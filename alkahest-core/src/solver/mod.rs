//! Polynomial system solver via Gröbner bases.
//!
//! # V2-11 — Regular chains
//!
//! [`regular_chains::triangularize`] exposes a Lex-basis triangular decomposition
//! (with optional bottom-variable factor splitting).  On a triangular back-sub
//! stall, [`solve_polynomial_system`] retries using an extracted regular chain
//! from the same Gröbner basis.
//!
//! # V2-14 — Homotopy continuation (numerical algebraic geometry)
//!
//! [`homotopy::solve_numerical`] runs a total-degree homotopy in `ℂⁿ` (Bézout
//! start system) and yields real roots with Smale-style checks and `ArbBall`
//! enclosures — see module documentation for limitations on **deficient**
//! systems.
//!
//! # V1-4 — Symbolic triangular solving (`solve_polynomial_system`)
//!
//! Inputs are polynomial equations (`lhs - rhs = 0`), variables, and an
//! `ExprPool`; outputs are symbolic `ExprId` values (may include `sqrt`),
//! or `SolutionSet::Parametric` / `SolutionSet::NoSolution`.
//!
//! Candidate tuples are checked against the input equations before they are
//! returned — see [`solve_polynomial_system`]'s post-condition and the
//! `verify` module.  Verifying is far cheaper than solving, and a returned
//! solution that does not satisfy the system is always a bug.
//!
//! Free symbols that appear in the equations but are not listed in `vars` are
//! treated as **parameters**: they become extra indeterminates in the Gröbner
//! basis (appended after the solve variables under Lex) and are pre-bound to
//! themselves during back-substitution, so solutions may involve those
//! symbols (e.g. `solve([x² − y], [x])` → `±√y`).

pub mod diophantine;
mod generic;
pub mod homotopy;
pub mod nsolve;
pub mod numeric;
mod opaque;
pub mod polyhedral;
mod rational;
pub mod regular_chains;
pub mod transcendental;
mod verify;

pub use transcendental::with_complex_domain;
pub use transcendental::{
    solve_transcendental, take_transcendental_refusal, TranscendentalOutcome,
};

pub use regular_chains::{
    extract_regular_chain_from_basis, main_variable_recursive, triangularize, RegularChain,
};

pub use homotopy::{
    solve_numerical, solve_numerical_complex, CertifiedPoint, HomotopyError, HomotopyOpts,
};
pub use nsolve::{
    nsolve, nsolve_all, NsolveError, NsolveStart, NumericRoot, RootIsolation, RootStatus,
};
pub use numeric::{numeric_solution_values, numeric_value, NumericSolveRefusal};

pub use diophantine::{
    diophantine, diophantine_pell_classes, DiophantineError, DiophantineSolution, PellClasses,
};

use crate::errors::AlkahestError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::poly::collect_free_vars;
use crate::poly::groebner::{GbPoly, GroebnerBasis, MonomialOrder};
use crate::poly::groebner::{ParamGbPoly, ParamPoly, QParam};
use rug::ops::Pow;
use rug::Rational;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single solution point: `ExprId`s parallel to the `vars` passed to the
/// solver.  Values may be rationals (for linear systems) or symbolic
/// expressions involving `sqrt` (for quadratic elimination tails).
pub type Solution = Vec<ExprId>;

/// The result of `solve_polynomial_system`.
pub enum SolutionSet {
    /// Finitely many solutions (each is a `Vec<ExprId>` parallel to `vars`).
    ///
    /// Every tuple has survived substitution back into the input equations, so
    /// a returned solution is never one the solver can itself refute.
    Finite(Vec<Solution>),
    /// **No finite solution list was produced**; the Gröbner basis is returned
    /// for downstream use.
    ///
    /// The usual reason is a positive-dimensional ideal.  It is also what the
    /// solver reports when the basis admits no complete triangular
    /// elimination in the declared unknowns, so this is "here is the ideal,
    /// enumerate it yourself" rather than a claim that solutions are infinite.
    Parametric(GroebnerBasis),
    /// No solution (ideal = ⟨1⟩).
    NoSolution,
}

/// Errors from the polynomial system solver.
#[derive(Debug, Clone)]
pub enum SolverError {
    /// An equation is not a polynomial (nor a rational function) in the given
    /// variables.
    NotPolynomial(String),
    /// Back-substitution reached a univariate of this degree (≥ 3) that it
    /// cannot solve exactly: its coefficients involve free parameters or
    /// algebraic numbers and an irreducible factor of this degree has no
    /// closed form. (Rational-coefficient univariates of any degree are
    /// solved, with `RootOf` for the factors that need it.)
    HighDegree(usize),
    /// Number of equations doesn't match number of variables (for zero-dim check).
    ShapeMismatch,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::NotPolynomial(s) => write!(f, "not a polynomial: {s}"),
            SolverError::HighDegree(d) => write!(
                f,
                "back-substitution requires solving a degree-{d} univariate polynomial \
                 whose coefficients are not rational numbers (free parameters or algebraic \
                 numbers), and it does not factor into pieces of degree ≤ 2"
            ),
            SolverError::ShapeMismatch => write!(
                f,
                "number of equations must equal number of variables for zero-dimensional solving"
            ),
        }
    }
}

impl std::error::Error for SolverError {}

impl AlkahestError for SolverError {
    fn code(&self) -> &'static str {
        match self {
            SolverError::NotPolynomial(_) => "E-SOLVE-001",
            SolverError::HighDegree(_) => "E-SOLVE-002",
            SolverError::ShapeMismatch => "E-SOLVE-003",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SolverError::NotPolynomial(_) => Some(
                "the unknowns must appear polynomially or rationally (a sub-expression free \
                 of them may be anything — it is a coefficient); a single equation with an \
                 unknown inside exp/log/sin/cos/tan is solved only in the closed-form shapes \
                 listed under `solve`",
            ),
            SolverError::HighDegree(_) => Some(
                "an irreducible factor of degree ≥ 3 with symbolic or algebraic coefficients \
                 has no closed form here; substitute values for the parameters, or retry with \
                 numeric=True or method=\"homotopy\"",
            ),
            SolverError::ShapeMismatch => {
                Some("provide one equation per variable for zero-dimensional system solving")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Everywhere-undefined equations, reported out of band
// ---------------------------------------------------------------------------

/// The solver declined because an equation's denominator is **identically**
/// zero — `1/(x − x)`.
///
/// Such an equation denotes no function, so it has no solution set at all;
/// clearing it would multiply through by zero and make every point a
/// "solution", which is the one outcome worth going out of the way to prevent.
///
/// # Why this is not an error variant
///
/// [`SolverError`] is a public *exhaustive* enum, so growing it a variant is a
/// major semver break for every downstream `match`. The refusal therefore
/// travels inside [`SolverError::NotPolynomial`] and is recovered here, exactly
/// as [`regular_chains::TriangularizeRefusal`] does for `E-SOLVE-004`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UndefinedEquation {
    detail: String,
}

impl UndefinedEquation {
    /// What was found to be identically zero.
    pub fn detail(&self) -> &str {
        &self.detail
    }
}

impl fmt::Display for UndefinedEquation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "equation is undefined everywhere: {}", self.detail)
    }
}

impl std::error::Error for UndefinedEquation {}

impl AlkahestError for UndefinedEquation {
    fn code(&self) -> &'static str {
        "E-SOLVE-005"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(
            "a denominator simplifies to zero, so the equation denotes no function; \
             check the equation for a subtraction that cancels",
        )
    }
}

thread_local! {
    /// The refusal behind the `SolverError::NotPolynomial` this thread is about
    /// to return, when that variant is a carrier rather than what it usually
    /// means.
    static LAST_UNDEFINED_EQUATION: std::cell::RefCell<Option<UndefinedEquation>> =
        const { std::cell::RefCell::new(None) };
}

/// Drop any recorded refusal, so a later unrelated `NotPolynomial` — a
/// genuinely transcendental equation, say — is never re-attributed to it.
fn forget_undefined_equation() {
    LAST_UNDEFINED_EQUATION.with(|c| *c.borrow_mut() = None);
}

/// Build the carrier error and stash the refusal behind it.
pub(crate) fn refuse_undefined_equation(detail: &str) -> SolverError {
    let refusal = UndefinedEquation {
        detail: detail.to_string(),
    };
    let message = refusal.to_string();
    LAST_UNDEFINED_EQUATION.with(|c| *c.borrow_mut() = Some(refusal));
    SolverError::NotPolynomial(message)
}

/// Take the refusal behind the error that just came back, if there was one.
///
/// Bindings call this when [`solve_polynomial_system`] returns
/// `SolverError::NotPolynomial` and raise the refusal's own `E-SOLVE-005` when
/// it is present, so the caller is told the equation is undefined rather than
/// merely non-polynomial. Consuming, so one refusal is reported once;
/// thread-local.
pub fn take_undefined_equation() -> Option<UndefinedEquation> {
    LAST_UNDEFINED_EQUATION.with(|c| c.borrow_mut().take())
}

/// Is a refusal waiting behind the `NotPolynomial` that just came back?
fn undefined_equation_pending() -> bool {
    LAST_UNDEFINED_EQUATION.with(|c| c.borrow().is_some())
}

// ---------------------------------------------------------------------------
// Expr → GbPoly conversion
// ---------------------------------------------------------------------------

/// Convert an `Expr` (which must be a polynomial in `vars`) to a `GbPoly`
/// with rational coefficients.  The variable order in the exponent vector
/// follows the order of `vars`.
///
/// Every exponent and every monomial's total degree of the result fits in
/// `u32`; an input that needs more (`x^(2^32)`, `x^(2^31)·x^(2^31)`,
/// `x^(2^31)·y^(2^31)`) is refused with `NotPolynomial` rather than wrapped —
/// it used to come back as a different polynomial (`solve` then reported no
/// solutions).
pub fn expr_to_gbpoly(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<GbPoly, SolverError> {
    let n = vars.len();
    expr_to_gbpoly_rec(expr, vars, n, pool)
}

/// The refusal for an exponent or total degree that does not fit in `u32`.
fn exponent_too_large() -> SolverError {
    SolverError::NotPolynomial(format!(
        "{} (E-POLY-004)",
        crate::poly::ConversionError::ExponentTooLarge
    ))
}

/// `a · b` for two `GbPoly`s with every product exponent and total degree
/// checked, unlike [`GbPoly::mul`] (which panics on overflow).
pub(super) fn gb_mul_checked(a: &GbPoly, b: &GbPoly) -> Result<GbPoly, SolverError> {
    let mut result = GbPoly::zero(a.n_vars);
    for (ea, ca) in &a.terms {
        for (eb, cb) in &b.terms {
            let e = crate::poly::exponent::add_vecs(ea, eb).map_err(|_| exponent_too_large())?;
            let entry = result.terms.entry(e).or_insert_with(|| Rational::from(0));
            *entry += Rational::from(ca * cb);
        }
    }
    result.terms.retain(|_, v| *v != 0);
    Ok(result)
}

/// `base^n` by repeated squaring, checked. The square after the top bit is
/// never formed: for `x^(2^31)` it would be `x^(2^32)`, which is refused even
/// though the power itself fits.
fn gb_pow_checked(base: GbPoly, mut n: u32) -> Result<GbPoly, SolverError> {
    let mut result = GbPoly::constant(Rational::from(1), base.n_vars);
    let mut cur = base;
    while n > 0 {
        if n & 1 == 1 {
            result = gb_mul_checked(&result, &cur)?;
        }
        n >>= 1;
        if n > 0 {
            cur = gb_mul_checked(&cur, &cur)?;
        }
    }
    Ok(result)
}

fn expr_to_gbpoly_rec(
    expr: ExprId,
    vars: &[ExprId],
    n_vars: usize,
    pool: &ExprPool,
) -> Result<GbPoly, SolverError> {
    if let Some(idx) = vars.iter().position(|&v| v == expr) {
        let mut exp = vec![0u32; n_vars];
        exp[idx] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exp, rug::Rational::from(1));
        return Ok(GbPoly { terms, n_vars });
    }

    enum Node {
        Var(usize),
        IntConst(rug::Integer),
        RatConst(Rational),
        FloatConst(f64),
        FreeSymbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String),
        Other,
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Integer(n) => Node::IntConst(n.0.clone()),
        ExprData::Rational(r) => Node::RatConst(r.0.clone()),
        ExprData::Float(f) => Node::FloatConst(f.inner.to_f64()),
        ExprData::Symbol { name, .. } => {
            if let Some(idx) = vars.iter().position(|&v| v == expr) {
                Node::Var(idx)
            } else {
                Node::FreeSymbol(name.clone())
            }
        }
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, .. } => Node::Func(name.clone()),
        _ => Node::Other,
    });

    match node {
        Node::Var(idx) => {
            let mut exp = vec![0u32; n_vars];
            exp[idx] = 1;
            let mut terms = BTreeMap::new();
            terms.insert(exp, Rational::from(1));
            Ok(GbPoly { terms, n_vars })
        }
        Node::IntConst(n) => Ok(GbPoly::constant(Rational::from(n), n_vars)),
        Node::RatConst(r) => Ok(GbPoly::constant(r, n_vars)),
        Node::FloatConst(v) => {
            let r = Rational::from_f64(v).unwrap_or_else(|| Rational::from(0));
            Ok(GbPoly::constant(r, n_vars))
        }
        Node::FreeSymbol(name) => Err(SolverError::NotPolynomial(format!(
            "free symbol '{name}' not in variable list"
        ))),
        Node::Add(args) => {
            let mut result = GbPoly::zero(n_vars);
            for a in args {
                let p = expr_to_gbpoly_rec(a, vars, n_vars, pool)?;
                result = result.add(&p);
            }
            Ok(result)
        }
        Node::Mul(args) => {
            let mut result = GbPoly::constant(Rational::from(1), n_vars);
            for a in args {
                let p = expr_to_gbpoly_rec(a, vars, n_vars, pool)?;
                result = gb_mul_checked(&result, &p)?;
            }
            Ok(result)
        }
        Node::Pow(base, exp_id) => {
            let exp_node = pool.with(exp_id, |d| match d {
                ExprData::Integer(n) => Some(n.0.clone()),
                _ => None,
            });
            match exp_node {
                Some(n) => {
                    if n < 0 {
                        return Err(SolverError::NotPolynomial(format!(
                            "negative exponent {n} in polynomial"
                        )));
                    }
                    let n_u32 = n.to_u32().ok_or_else(exponent_too_large)?;
                    let base_poly = expr_to_gbpoly_rec(base, vars, n_vars, pool)?;
                    gb_pow_checked(base_poly, n_u32)
                }
                None => Err(SolverError::NotPolynomial(
                    "symbolic or non-integer exponent".to_string(),
                )),
            }
        }
        Node::Func(name) => Err(SolverError::NotPolynomial(format!(
            "function '{name}' is not a polynomial"
        ))),
        Node::Other => Err(SolverError::NotPolynomial(
            "unsupported expression node".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Expr → ParamGbPoly conversion (M9)
// ---------------------------------------------------------------------------

/// Convert an `Expr` to a [`ParamGbPoly`] over `Q(params)[vars]`.
///
/// The expression must be *polynomial in `vars`* and *rational in `params`*.
/// That second half is the difference from [`expr_to_gbpoly`], which refuses
/// any negative exponent: here a negative power whose base is free of `vars` is
/// just a denominator in the coefficient field, which is exactly where the
/// parameters live.  Without it a parametric basis cannot be fed its own
/// [`crate::poly::groebner::ParamGroebnerBasis::generators`] back — those carry
/// `den^-1` factors by construction.
///
/// Exponent slot `i` names `vars[i]`; parameter slot `j` names `params[j]`.
/// `vars` and `params` must be disjoint; a symbol in neither list is an error,
/// as it is for [`expr_to_gbpoly`].
pub fn expr_to_param_gbpoly(
    expr: ExprId,
    vars: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> Result<ParamGbPoly, SolverError> {
    // The ℚ(params) arithmetic below is infallible (and panics rather than
    // wrap on exponent overflow), so bound every degree it can reach first.
    if param_degree_bound(expr, vars, params, pool, &mut BTreeMap::new()) > u64::from(u32::MAX) {
        return Err(exponent_too_large());
    }
    expr_to_param_gbpoly_rec(expr, vars, params, pool)
}

/// An upper bound, saturating, on every exponent and total degree — in the
/// ring variables and in the parameters, numerators and denominators alike —
/// that converting `expr` can produce. Sums are bounded by the *sum* of their
/// terms' bounds, because adding fractions of `Q(params)` multiplies their
/// denominators.
fn param_degree_bound(
    expr: ExprId,
    vars: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
    memo: &mut BTreeMap<ExprId, u64>,
) -> u64 {
    if vars.contains(&expr) || params.contains(&expr) {
        return 1;
    }
    if let Some(&b) = memo.get(&expr) {
        return b;
    }
    enum Shape {
        Leaf,
        Sum(Vec<ExprId>),
        Pow(ExprId, ExprId),
    }
    let shape = pool.with(expr, |d| match d {
        ExprData::Add(args) | ExprData::Mul(args) => Shape::Sum(args.clone()),
        ExprData::Pow { base, exp } => Shape::Pow(*base, *exp),
        _ => Shape::Leaf,
    });
    let b = match shape {
        Shape::Leaf => 0,
        Shape::Sum(args) => args.into_iter().fold(0u64, |acc, a| {
            acc.saturating_add(param_degree_bound(a, vars, params, pool, memo))
        }),
        Shape::Pow(base, exp) => {
            // A non-integer exponent is refused by the conversion itself.
            let k = pool.with(exp, |e| match e {
                ExprData::Integer(n) => n.0.clone().abs().to_u64().unwrap_or(u64::MAX),
                _ => 0,
            });
            param_degree_bound(base, vars, params, pool, memo).saturating_mul(k)
        }
    };
    memo.insert(expr, b);
    b
}

fn param_constant(c: QParam, n_vars: usize, n_params: usize) -> ParamGbPoly {
    let mut p = ParamGbPoly::zero(n_vars, n_params);
    if !c.is_zero() {
        p.terms.insert(vec![0u32; n_vars], c);
    }
    p
}

fn expr_to_param_gbpoly_rec(
    expr: ExprId,
    vars: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> Result<ParamGbPoly, SolverError> {
    let (n_vars, n_params) = (vars.len(), params.len());
    if let Some(idx) = vars.iter().position(|&v| v == expr) {
        let mut exp = vec![0u32; n_vars];
        exp[idx] = 1;
        let mut p = ParamGbPoly::zero(n_vars, n_params);
        p.terms.insert(exp, QParam::one(n_params));
        return Ok(p);
    }
    if let Some(idx) = params.iter().position(|&p| p == expr) {
        let c = QParam::from_poly(ParamPoly::var(idx, n_params));
        return Ok(param_constant(c, n_vars, n_params));
    }

    enum Node {
        IntConst(rug::Integer),
        RatConst(Rational),
        FloatConst(f64),
        FreeSymbol(String),
        Add(Vec<ExprId>),
        Mul(Vec<ExprId>),
        Pow(ExprId, ExprId),
        Func(String),
        Other,
    }

    let node = pool.with(expr, |data| match data {
        ExprData::Integer(n) => Node::IntConst(n.0.clone()),
        ExprData::Rational(r) => Node::RatConst(r.0.clone()),
        ExprData::Float(f) => Node::FloatConst(f.inner.to_f64()),
        ExprData::Symbol { name, .. } => Node::FreeSymbol(name.clone()),
        ExprData::Add(args) => Node::Add(args.clone()),
        ExprData::Mul(args) => Node::Mul(args.clone()),
        ExprData::Pow { base, exp } => Node::Pow(*base, *exp),
        ExprData::Func { name, .. } => Node::Func(name.clone()),
        _ => Node::Other,
    });

    let rational_const = |r: Rational| {
        Ok(param_constant(
            QParam::from_rational(&r, n_params),
            n_vars,
            n_params,
        ))
    };

    match node {
        Node::IntConst(n) => rational_const(Rational::from(n)),
        Node::RatConst(r) => rational_const(r),
        Node::FloatConst(v) => rational_const(Rational::from_f64(v).unwrap_or_else(|| 0.into())),
        Node::FreeSymbol(name) => Err(SolverError::NotPolynomial(format!(
            "free symbol '{name}' is neither a ring variable nor a parameter"
        ))),
        Node::Add(args) => {
            let mut acc = ParamGbPoly::zero(n_vars, n_params);
            for a in args {
                acc = acc.add(&expr_to_param_gbpoly_rec(a, vars, params, pool)?);
            }
            Ok(acc)
        }
        Node::Mul(args) => {
            let mut acc = param_constant(QParam::one(n_params), n_vars, n_params);
            for a in args {
                acc = acc.mul(&expr_to_param_gbpoly_rec(a, vars, params, pool)?);
            }
            Ok(acc)
        }
        Node::Pow(base, exp_id) => {
            let exp_node = pool.with(exp_id, |d| match d {
                ExprData::Integer(n) => n.0.to_i64(),
                _ => None,
            });
            let Some(n_val) = exp_node else {
                return Err(SolverError::NotPolynomial(
                    "symbolic or non-integer exponent".to_string(),
                ));
            };
            let base_poly = expr_to_param_gbpoly_rec(base, vars, params, pool)?;
            let (base_poly, k) = if n_val < 0 {
                // A denominator is only admissible in the coefficient field.
                let Some(c) = base_poly.as_coeff() else {
                    return Err(SolverError::NotPolynomial(format!(
                        "negative exponent {n_val} on a ring variable; only the \
                         coefficient field Q(params) admits denominators"
                    )));
                };
                let Some(inv) = c.inv() else {
                    return Err(SolverError::NotPolynomial(
                        "negative exponent on zero".to_string(),
                    ));
                };
                (param_constant(inv, n_vars, n_params), n_val.unsigned_abs())
            } else {
                (base_poly, n_val as u64)
            };
            let mut result = param_constant(QParam::one(n_params), n_vars, n_params);
            let mut cur = base_poly;
            let mut rem = k;
            while rem > 0 {
                if rem & 1 == 1 {
                    result = result.mul(&cur);
                }
                rem >>= 1;
                // No square past the top bit: it can overflow when the power
                // itself does not.
                if rem > 0 {
                    let cur2 = cur.clone();
                    cur = cur.mul(&cur2);
                }
            }
            Ok(result)
        }
        Node::Func(name) => Err(SolverError::NotPolynomial(format!(
            "function '{name}' is not a polynomial"
        ))),
        Node::Other => Err(SolverError::NotPolynomial(
            "unsupported expression node".to_string(),
        )),
    }
}

// ---------------------------------------------------------------------------
// GbPoly → Expr conversion
// ---------------------------------------------------------------------------

/// Rebuild an `Expr` from a [`GbPoly`] — the inverse of [`expr_to_gbpoly`].
///
/// `vars` must be the same variable list, in the same order, that produced the
/// polynomial's exponent vectors: exponent slot `i` names `vars[i]`.
///
/// Returns `None` when `vars` is too short to name every variable the
/// polynomial actually uses; silently mis-naming exponent slots would be worse
/// than refusing.  The zero polynomial converts to the integer `0`.
pub fn gbpoly_to_expr(poly: &GbPoly, vars: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut terms: Vec<ExprId> = Vec::with_capacity(poly.terms.len());
    for (exp, coeff) in &poly.terms {
        if *coeff == 0 {
            continue;
        }
        let mut factors: Vec<ExprId> = Vec::new();
        for (i, &e) in exp.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let v = *vars.get(i)?;
            factors.push(if e == 1 {
                v
            } else {
                pool.pow(v, pool.integer(e))
            });
        }
        // Keep an explicit coefficient factor unless it is a bare `1` in front
        // of at least one variable.
        if factors.is_empty() || *coeff != 1 {
            factors.insert(0, rational_to_expr(coeff, pool));
        }
        terms.push(if factors.len() == 1 {
            factors[0]
        } else {
            pool.mul(factors)
        });
    }
    Some(match terms.len() {
        0 => pool.integer(0),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

// ---------------------------------------------------------------------------
// ExprId builders
// ---------------------------------------------------------------------------

fn rational_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    let (num, den) = r.clone().into_numer_denom();
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

fn neg_expr(e: ExprId, pool: &ExprPool) -> ExprId {
    let neg_one = pool.integer(rug::Integer::from(-1));
    pool.mul(vec![neg_one, e])
}

fn div_expr(num: ExprId, den: ExprId, pool: &ExprPool) -> ExprId {
    // num / den = num * den^(-1)
    let neg_one = pool.integer(rug::Integer::from(-1));
    let inv_den = pool.pow(den, neg_one);
    pool.mul(vec![num, inv_den])
}

/// Is this `ExprId` certainly zero, by structure or by exact rational value?
fn is_zero_value(e: ExprId, pool: &ExprPool) -> bool {
    is_certain_zero(e, pool) || rational_value(e, pool).is_some_and(|v| v == 0)
}

/// Exact rational value of `expr`, or `None` when it is not a rational
/// arithmetic expression (a radical, a parameter, a division by zero).
///
/// The expression pool does not fold arithmetic on literals — `0 · 4 · 1` and
/// `(−2)²` both survive as nodes — so a vanishing discriminant reaches
/// [`solve_univariate_symbolic`] unrecognisable by structure alone.  Evaluating
/// the handful of node kinds the solver builds costs nothing and decides it
/// exactly, which is what turns `±√0/2` back into the single root it is.
fn rational_value(expr: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(expr) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        ExprData::Add(args) => args.iter().try_fold(Rational::from(0), |acc, &a| {
            Some(acc + rational_value(a, pool)?)
        }),
        ExprData::Mul(args) => args.iter().try_fold(Rational::from(1), |acc, &a| {
            Some(acc * rational_value(a, pool)?)
        }),
        ExprData::Pow { base, exp } => {
            let ExprData::Integer(k) = pool.get(exp) else {
                return None;
            };
            let k = k.0.to_i32()?;
            let b = rational_value(base, pool)?;
            if k < 0 && b == 0 {
                return None;
            }
            Some(b.pow(k))
        }
        _ => None,
    }
}

/// Is this `ExprId` **certainly** zero?
///
/// Recognises the shapes back-substitution actually produces without invoking
/// the simplifier: a literal zero, a sum of zeros, `√0`, and `0^k` for `k > 0`.
/// The last two matter because a vanishing discriminant arrives as `0² + 0`
/// rather than as `0`, and a plain literal test then reported the double root
/// of `x² = 0` as the two entries `±√0/2`.
///
/// One-sided by design: `false` means "not recognised as zero", never "known
/// non-zero".  Products are deliberately not folded — a zero factor does not
/// make `0 · 0⁻¹` zero.
fn is_certain_zero(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 == 0,
        ExprData::Rational(r) => r.0 == 0,
        ExprData::Add(args) => args.iter().all(|&a| is_certain_zero(a, pool)),
        ExprData::Pow { base, exp } => {
            let positive = matches!(pool.get(exp), ExprData::Integer(k) if k.0 > 0);
            positive && is_certain_zero(base, pool)
        }
        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
            is_certain_zero(args[0], pool)
        }
        _ => false,
    }
}

/// Extract the coefficient of `var_idx^k` in `poly`, substituting
/// already-solved vars (`assigned[i] = Some(ExprId)`) into the remaining
/// factors.  Unsolved vars that happen to appear (other than `var_idx`)
/// are left as their original `ExprId` variable — callers should only
/// invoke this when the generator involves exactly one unsolved variable
/// at `var_idx`.
fn extract_coeff_in_var(
    poly: &GbPoly,
    var_idx: usize,
    k: u32,
    vars: &[ExprId],
    assigned: &[Option<ExprId>],
    pool: &ExprPool,
) -> ExprId {
    let mut sum_terms: Vec<ExprId> = Vec::new();
    for (exp, coeff) in &poly.terms {
        let e_k = exp.get(var_idx).copied().unwrap_or(0);
        if e_k != k {
            continue;
        }
        let mut factors: Vec<ExprId> = Vec::new();
        if *coeff != 1 {
            factors.push(rational_to_expr(coeff, pool));
        }
        for (i, &e) in exp.iter().enumerate() {
            if i == var_idx || e == 0 {
                continue;
            }
            let base = assigned
                .get(i)
                .and_then(|o| o.as_ref())
                .copied()
                .unwrap_or(vars[i]);
            if e == 1 {
                factors.push(base);
            } else {
                let exp_id = pool.integer(rug::Integer::from(e));
                factors.push(pool.pow(base, exp_id));
            }
        }
        let term = match factors.len() {
            0 => pool.integer(rug::Integer::from(1)),
            1 => factors[0],
            _ => pool.mul(factors),
        };
        // Re-apply the rational coefficient sign if it wasn't a 1 above
        let signed = if *coeff == 1 {
            term
        } else {
            // Already included in factors
            term
        };
        sum_terms.push(signed);
    }
    match sum_terms.len() {
        0 => pool.integer(rug::Integer::from(0)),
        1 => sum_terms[0],
        _ => pool.add(sum_terms),
    }
}

// ---------------------------------------------------------------------------
// Univariate solver (symbolic output, ℚ-only and symbolic paths)
// ---------------------------------------------------------------------------

/// Solve `a₀ + a₁·x + a₂·x² = 0` where each `aᵢ` is an already-substituted
/// `ExprId`.  Returns a `Vec<ExprId>` of roots (symbolic).  Degree is
/// inferred from `coeffs.len()`; higher-degree terms must be syntactic-zero
/// (the caller trims first).
///
/// A degree-2 equation yields **one** root when the discriminant collapses to
/// a syntactic zero and two otherwise.  `x² = 0` has the solution *set* `{0}`;
/// reporting `±√0/2` as two entries was a wrong count, not a multiplicity
/// annotation, and it multiplied across variables (`[x², y², z²]` reported
/// eight copies of the origin).  Roots that coincide for a subtler reason are
/// collapsed later by the numeric de-duplication in [`refine_solutions`].
fn solve_univariate_symbolic(
    coeffs: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<ExprId>, SolverError> {
    let mut degree = 0usize;
    for (i, &c) in coeffs.iter().enumerate() {
        if !is_zero_value(c, pool) {
            degree = i;
        }
    }
    match degree {
        0 => {
            // Constant equation.  If coefficient is zero it's trivially
            // satisfied (0 = 0) — shouldn't happen for a proper generator.
            // Otherwise it's 0 = nonzero → no solution, but we signal that
            // by returning empty (the caller treats this as contradiction).
            Ok(vec![])
        }
        1 => {
            let a = coeffs[1];
            let b = coeffs[0];
            let neg_b = neg_expr(b, pool);
            Ok(vec![div_expr(neg_b, a, pool)])
        }
        2 => {
            let a = coeffs[2];
            let b = coeffs[1];
            let c = coeffs[0];
            let two = pool.integer(rug::Integer::from(2));
            let four = pool.integer(rug::Integer::from(4));
            let b2 = pool.pow(b, two);
            let four_ac = pool.mul(vec![four, a, c]);
            let neg_four_ac = neg_expr(four_ac, pool);
            let disc = pool.add(vec![b2, neg_four_ac]);
            let two_b = pool.integer(rug::Integer::from(2));
            let two_a = pool.mul(vec![two_b, a]);
            let neg_b = neg_expr(b, pool);
            if is_zero_value(disc, pool) {
                return Ok(vec![div_expr(neg_b, two_a, pool)]);
            }
            let sqrt_disc = pool.func("sqrt", vec![disc]);
            let root_plus = div_expr(pool.add(vec![neg_b, sqrt_disc]), two_a, pool);
            let neg_sqrt = neg_expr(sqrt_disc, pool);
            let root_minus = div_expr(pool.add(vec![neg_b, neg_sqrt]), two_a, pool);
            Ok(vec![root_plus, root_minus])
        }
        d => Err(SolverError::HighDegree(d)),
    }
}

// ---------------------------------------------------------------------------
// Main solver
// ---------------------------------------------------------------------------

/// Highest power of `var_idx` occurring in `poly`.
fn max_degree_in_var(poly: &GbPoly, var_idx: usize) -> u32 {
    poly.terms
        .keys()
        .map(|e| e.get(var_idx).copied().unwrap_or(0))
        .max()
        .unwrap_or(0)
}

/// Solve-variable indices that occur in `poly` (parameters are ignored: they
/// are pre-bound and never block a step).
fn active_solve_vars(poly: &GbPoly, n_solve: usize) -> Vec<usize> {
    (0..n_solve)
        .filter(|&i| {
            poly.terms
                .keys()
                .any(|e| e.get(i).copied().unwrap_or(0) > 0)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Assumed hypotheses, reported out of band
// ---------------------------------------------------------------------------

thread_local! {
    /// Leading coefficients the back-solver divided by without being able to
    /// prove them non-zero, for the [`solve_polynomial_system`] call in
    /// progress. De-duplicated, in the order they were assumed.
    static ASSUMED_NONZERO: std::cell::RefCell<Vec<ExprId>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

thread_local! {
    /// Hypotheses of other shapes — `c > 0` for a real logarithm, `k ∈ ℤ` for
    /// the index of a periodic family — recorded by the transcendental path.
    static ASSUMED_OTHER: std::cell::RefCell<Vec<crate::deriv::log::SideCondition>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record a hypothesis that is not a plain `≠ 0`.
pub(crate) fn record_side_condition(cond: crate::deriv::log::SideCondition) {
    ASSUMED_OTHER.with(|c| {
        let mut v = c.borrow_mut();
        if !v.contains(&cond) {
            v.push(cond);
        }
    });
}

/// Record that the solver divided by `lead` without deciding it is non-zero.
fn assume_nonzero(lead: ExprId) {
    ASSUMED_NONZERO.with(|c| {
        let mut v = c.borrow_mut();
        if !v.contains(&lead) {
            v.push(lead);
        }
    });
}

/// The hypotheses the solutions from the most recent [`solve_polynomial_system`]
/// call on this thread rest on, as [`crate::deriv::SideCondition::NonZero`].
///
/// `solve([a·x − b], [x])` returns `b/a`, which is the answer **for `a ≠ 0`**:
/// at `a = 0` the equation is `−b = 0`, so there is either no solution (`b ≠ 0`)
/// or every `x` (`b = 0`), and neither is `b/a`. The generic-parameter reading
/// is a deliberate and useful one, but a caller cannot audit an assumption that
/// is never stated. A parametric tuple is checked as a function of the
/// parameters (its residual must be identically zero), which says it is a
/// solution *generically*; where it stops being one is exactly what is listed
/// here — the hypotheses of the branches that were kept, and only those.
///
/// # Why out of band
///
/// [`SolutionSet`] is a public *exhaustive* enum and `solve_polynomial_system`'s
/// return type is public, so neither can grow a conditions field without a major
/// semver break. The hypotheses therefore travel beside the result, in the shape
/// `DerivedResult.verification["side_conditions"]` already uses — the same
/// treatment `zeilberger`'s natural-boundary hypothesis was given, and the same
/// out-of-band channel as [`crate::matrix::take_zero_test_refusal`].
///
/// Consuming, so one call's hypotheses cannot be read as a later call's. Empty
/// means the solver proved every coefficient it divided by to be non-zero — not
/// that it did not look.
///
/// The transcendental path ([`solve_transcendental`]) reports here too, with
/// two more shapes: [`SideCondition::Positive`](crate::deriv::log::SideCondition::Positive)
/// for the argument of a real logarithm it could not prove positive
/// (`log(kₑ/kₐ)` needs `kₑ/kₐ > 0`), and
/// [`SideCondition::InDomain`](crate::deriv::log::SideCondition::InDomain)`(k, Integer)`
/// for the index of a periodic solution family (`x = asin(c) + 2πk`).
pub fn take_solve_side_conditions() -> Vec<crate::deriv::log::SideCondition> {
    let mut out: Vec<crate::deriv::log::SideCondition> = ASSUMED_NONZERO.with(|c| {
        std::mem::take(&mut *c.borrow_mut())
            .into_iter()
            .map(crate::deriv::log::SideCondition::NonZero)
            .collect()
    });
    out.extend(ASSUMED_OTHER.with(|c| std::mem::take(&mut *c.borrow_mut())));
    out
}

/// Can the degree-`d` coefficient be relied on to be non-zero at this partial
/// assignment?
///
/// This is the property that makes one back-substitution step *complete*: if
/// the leading coefficient does not vanish, the substituted generator really
/// has degree `d` in the unknown and the quadratic formula returns **all** of
/// its roots.  When it does vanish, the same formula divides by zero and the
/// branch's true roots disappear — which is how `⟨x² + 3y, 2xy + 3x⟩` lost
/// `(±3/√2, −3/2)`: the chosen generator's leading coefficient was `2y + 3`,
/// zero on exactly the branch `y = −3/2`.
///
/// A coefficient still mentioning a free parameter is accepted, preserving the
/// documented generic-parameter reading of `solve([a·x − b], [x]) → b/a` — but
/// it is accepted as an **assumption**, recorded through [`assume_nonzero`] and
/// reported by [`take_solve_side_conditions`]. The reading is only defensible
/// while the caller can see what was assumed: `b/a` is the solution for `a ≠ 0`
/// and is wrong at `a = 0`, where the system has no solution, or every `x`.
fn leading_is_reliable(
    lead: ExprId,
    vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> LeadStatus {
    if let Some(v) = rational_value(lead, pool) {
        return if v != 0 {
            LeadStatus::Nonzero
        } else {
            LeadStatus::Unusable
        };
    }
    match verify::CBallEval::default().eval(lead, pool) {
        Ok(ball) => {
            if ball.excludes_zero() {
                LeadStatus::Nonzero
            } else {
                LeadStatus::Unusable
            }
        }
        // `Unsupported` is the parametric case, decided as a function of the
        // parameters: a coefficient that is identically zero at this partial
        // assignment (`b·y − a` at `y = a/b`, `d·0`) divides nothing, and
        // treating it as an assumption is what produced `x = 0·0⁻¹`.
        Err(verify::VerifyGap::Unsupported) if n_solve < vars.len() => {
            match generic::status(lead, vars, n_solve, pool) {
                generic::Generic::Nonzero(Some(numer)) => {
                    let hyps = generic::nonzero_hypotheses(&numer, vars, pool);
                    if hyps.is_empty() {
                        LeadStatus::Nonzero
                    } else {
                        LeadStatus::AssumedNonzero(hyps)
                    }
                }
                generic::Generic::Nonzero(None) => LeadStatus::AssumedNonzero(vec![lead]),
                generic::Generic::Zero
                | generic::Generic::Undefined
                | generic::Generic::Unknown => LeadStatus::Unusable,
            }
        }
        Err(verify::VerifyGap::Unsupported) => LeadStatus::AssumedNonzero(vec![lead]),
        Err(verify::VerifyGap::Undefined) => LeadStatus::Unusable,
    }
}

/// What [`leading_is_reliable`] could establish about a leading coefficient.
#[derive(Clone, PartialEq, Eq, Debug)]
enum LeadStatus {
    /// Proved non-zero — the step is unconditional.
    Nonzero,
    /// Not the zero function of the parameters, but zero for some of their
    /// values: usable only under the hypothesis that these factors do not
    /// vanish, which the caller must be told about.
    AssumedNonzero(Vec<ExprId>),
    /// Zero, or not a usable coefficient under any reading.
    Unusable,
}

/// One back-substitution step for one partial assignment: which unknown to
/// solve next, and the coefficients of the univariate it satisfies.
///
/// A generator is usable when every solve variable it mentions except the
/// chosen one is already assigned, and its leading coefficient in that unknown
/// survives [`leading_is_reliable`].  Nothing here depends on the elimination
/// order matching the monomial order, which matters: a Lex basis such as
/// `⟨x² − 2, x·y − y², y³ − 2y⟩` is only tractable by eliminating `x` first —
/// insisting on the Lex-last unknown reaches the cubic `y³ − 2y` and refuses a
/// system that is perfectly within scope.
///
/// `Err(HighDegree)` is reserved for the case where the *only* obstruction is
/// a degree above 2, which keeps `E-SOLVE-002` meaning what it documents.
fn find_step(
    gens: &[GbPoly],
    partial: &[Option<ExprId>],
    vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Result<Option<Step>, SolverError> {
    let mut best: Option<(Step, u32)> = None;
    let mut blocked_by_degree: Option<u32> = None;

    for (gi, g) in gens.iter().enumerate() {
        let active = active_solve_vars(g, n_solve);
        let unassigned: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&i| partial[i].is_none())
            .collect();
        let [var_idx] = unassigned[..] else {
            continue;
        };
        let deg = max_degree_in_var(g, var_idx);
        if deg == 0 {
            continue;
        }
        if best.as_ref().is_some_and(|(_, bd)| *bd <= deg) {
            continue;
        }
        let coeffs: Vec<ExprId> = (0..=deg)
            .map(|k| extract_coeff_in_var(g, var_idx, k, vars, partial, pool))
            .collect();
        // Above degree 2 there is no formula to apply blindly; the univariate
        // is factored instead, which needs coefficients that are rational
        // numbers, or a generator that is a polynomial in this unknown and the
        // free parameters alone (so it can be factored over ℚ[params]).
        let kind = if deg <= 2 {
            StepKind::Formula
        } else if coeffs.iter().all(|&c| rational_value(c, pool).is_some()) {
            StepKind::Rational
        } else if active == [var_idx] {
            StepKind::Parametric(gi)
        } else {
            blocked_by_degree = Some(blocked_by_degree.map_or(deg, |d: u32| d.min(deg)));
            continue;
        };
        let lead = coeffs[deg as usize];
        let hyps = match leading_is_reliable(lead, vars, n_solve, pool) {
            LeadStatus::Unusable => continue,
            LeadStatus::Nonzero => Vec::new(),
            LeadStatus::AssumedNonzero(h) => h,
        };
        best = Some((
            Step {
                var_idx,
                coeffs,
                kind,
                hyps,
            },
            deg,
        ));
    }

    match best {
        // Only the step actually taken contributes a hypothesis (generators
        // that were examined and passed over divide nothing), and only to the
        // branch it was taken on: see [`try_backsolve_generators`].
        Some((step, _)) => Ok(Some(step)),
        None => match blocked_by_degree {
            Some(d) => Err(SolverError::HighDegree(d as usize)),
            None => Ok(None),
        },
    }
}

/// One back-substitution step: the unknown to solve for, the coefficients
/// (`x^0 … x^d`) of the univariate it satisfies at the current partial
/// assignment, and how that univariate is solved.
struct Step {
    var_idx: usize,
    coeffs: Vec<ExprId>,
    kind: StepKind,
    /// What dividing by the leading coefficient assumed (`≠ 0`) on this branch.
    hyps: Vec<ExprId>,
}

/// How a [`Step`]'s univariate is solved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StepKind {
    /// Degree ≤ 2: the linear / quadratic formula on the coefficients as given.
    Formula,
    /// Degree ≥ 3 with rational-number coefficients: factor over ℚ, then each
    /// factor by formula, radical reduction, or `RootOf`.
    Rational,
    /// Degree ≥ 3, and the generator (index into the generator list) is a
    /// polynomial in this unknown and the free parameters only: factor over
    /// ℚ[params], then each factor by formula.
    Parametric(usize),
}

/// Solve one step's univariate. Every distinct root is returned, once.
fn solve_step(
    step: &Step,
    gens: &[GbPoly],
    vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Result<Vec<ExprId>, SolverError> {
    match step.kind {
        StepKind::Formula => {
            if n_solve < vars.len() {
                if let Some(roots) = parametric_formula_roots(step, vars, pool) {
                    return Ok(roots);
                }
            }
            solve_univariate_symbolic(&step.coeffs, pool)
        }
        StepKind::Rational => {
            let rats: Vec<Rational> = step
                .coeffs
                .iter()
                .map(|&c| rational_value(c, pool).expect("checked rational in find_step"))
                .collect();
            let ints = clear_denominators(&rats);
            crate::poly::univariate_roots::roots_over_q(&ints, pool)
                .map(|v| v.into_iter().map(|(r, _)| r).collect())
                .map_err(SolverError::HighDegree)
        }
        StepKind::Parametric(gi) => {
            let mp = gbpoly_to_multipoly(&gens[gi], vars);
            crate::poly::univariate_roots::roots_with_parameters(&mp, step.var_idx, pool)
                .map(|v| v.into_iter().map(|(r, _)| r).collect())
                .map_err(SolverError::HighDegree)
        }
    }
}

/// The roots of a degree-≤ 2 step whose coefficients are rational functions of
/// the parameters, found by **factoring** rather than by formula.
///
/// The quadratic formula applied to `b·c·y² − a·c·y` (a perfect-square
/// discriminant) returns `(a·c + √(a²c²))/(2bc)`: correct as a set, but a
/// branch-dependent radical that every later step has to carry — and that
/// [`leading_is_reliable`] cannot recognise as `a/b`, so the coefficient
/// `b·y − a` it makes vanish was being treated as non-zero.  Clearing
/// denominators and factoring over ℚ[x, params] returns `0` and `a/b` instead.
///
/// `None` when a coefficient is not a rational function of the parameters (a
/// radical from an earlier step); the caller then uses the formula.
fn parametric_formula_roots(step: &Step, vars: &[ExprId], pool: &ExprPool) -> Option<Vec<ExprId>> {
    let x = vars[step.var_idx];
    let terms: Vec<ExprId> = step
        .coeffs
        .iter()
        .enumerate()
        .map(|(k, &c)| match k {
            0 => c,
            1 => pool.mul(vec![c, x]),
            _ => pool.mul(vec![c, pool.pow(x, pool.integer(k as u32))]),
        })
        .collect();
    let p = if terms.len() == 1 {
        terms[0]
    } else {
        pool.add(terms)
    };
    let cleared = rational::clear_denominators(p, vars, pool).ok()?;
    if cleared.numer.is_zero() {
        return None;
    }
    let mp = gbpoly_to_multipoly(&cleared.numer, vars);
    crate::poly::univariate_roots::roots_with_parameters(&mp, step.var_idx, pool)
        .ok()
        .map(|v| v.into_iter().map(|(r, _)| r).collect())
}

/// Integer coefficients with the same roots: multiply through by the lcm of
/// the denominators.
fn clear_denominators(rats: &[Rational]) -> Vec<rug::Integer> {
    let mut l = rug::Integer::from(1);
    for r in rats {
        l.lcm_mut(r.denom());
    }
    rats.iter()
        .map(|r| {
            let v = Rational::from(r * &l);
            v.numer().clone()
        })
        .collect()
}

/// A Gröbner-ring polynomial over ℚ as an integer [`MultiPoly`] over `vars`
/// (the same indeterminates, in the same order), denominators cleared.
///
/// [`MultiPoly`]: crate::poly::MultiPoly
fn gbpoly_to_multipoly(g: &GbPoly, vars: &[ExprId]) -> crate::poly::MultiPoly {
    let rats: Vec<Rational> = g.terms.values().cloned().collect();
    let ints = clear_denominators(&rats);
    let mut terms = std::collections::BTreeMap::new();
    for (exp, c) in g.terms.keys().zip(ints) {
        let mut e: Vec<u32> = exp.clone();
        while e.last() == Some(&0) {
            e.pop();
        }
        terms.insert(e, c);
    }
    crate::poly::MultiPoly {
        vars: vars.to_vec(),
        terms,
    }
}

/// A candidate solution and the hypotheses (`≠ 0`) its branch divided by.
///
/// The hypotheses travel with the branch, not with the call: a branch that is
/// later refuted or found undefined must not leave its assumptions behind in
/// [`take_solve_side_conditions`] — which is how `solve_side_conditions()`
/// came to list `(d·0) ≠ 0`.
type Candidate = (Solution, Vec<ExprId>);

/// Backsolve over a fixed generator list (full Gröbner basis or a triangular
/// subset).
enum BacksolveOutcome {
    Finite(Vec<Candidate>),
    /// Some branch reached a point where no generator determines a remaining
    /// unknown — caller may retry a smaller set.
    Stuck,
    NoSolution,
}

/// Backsolve over a fixed generator list.
///
/// `vars` is the full indeterminate list (solve unknowns first, then free
/// parameters).  `n_solve` is the number of unknowns to assign; indices
/// `n_solve..vars.len()` are pre-bound to themselves (parametric coefficients).
///
/// Each branch picks its own next step (see [`find_step`]), so the candidate
/// set it produces contains every solution of the ideal that the branch's
/// partial assignment is consistent with.  Filtering the union back down to
/// the true solutions is [`refine_solutions`]' job.
fn try_backsolve_generators(
    gens: &[GbPoly],
    vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Result<BacksolveOutcome, SolverError> {
    let n_vars = vars.len();
    debug_assert!(n_solve <= n_vars);

    let mut initial = vec![None; n_vars];
    for i in n_solve..n_vars {
        initial[i] = Some(vars[i]);
    }
    let mut partials: Vec<(Vec<Option<ExprId>>, Vec<ExprId>)> = vec![(initial, Vec::new())];

    for _ in 0..n_solve {
        let mut new_partials = Vec::new();
        let mut high_degree: Option<SolverError> = None;
        for (partial, hyps) in &partials {
            if crate::budget::interrupted() {
                return Err(interrupted_error());
            }
            let step = match find_step(gens, partial, vars, n_solve, pool) {
                Ok(s) => s,
                // A degree-blocked branch does not end the level on its own:
                // another branch may turn out to be under-determined, and that
                // is the refusal worth reporting.  If nothing worse turns up,
                // the whole solve declines with `E-SOLVE-002` — returning the
                // branches that *did* resolve would be an incomplete solution
                // set presented as a complete one.
                Err(e) => {
                    high_degree = Some(e);
                    continue;
                }
            };
            let Some(step) = step else {
                if partial_is_refuted(gens, partial, vars, n_solve, pool) {
                    // A dead branch, not an under-determined one: drop it.
                    continue;
                }
                return Ok(BacksolveOutcome::Stuck);
            };
            for root in solve_step(&step, gens, vars, n_solve, pool)? {
                let mut np = partial.clone();
                np[step.var_idx] = Some(root);
                let mut nh = hyps.clone();
                for &h in &step.hyps {
                    if !nh.contains(&h) {
                        nh.push(h);
                    }
                }
                new_partials.push((np, nh));
            }
        }
        if let Some(e) = high_degree {
            return Err(e);
        }
        partials = new_partials;
        if partials.is_empty() {
            return Ok(BacksolveOutcome::NoSolution);
        }
    }

    let solutions: Vec<Candidate> = partials
        .into_iter()
        .map(|(p, h)| {
            let sol = p
                .into_iter()
                .take(n_solve)
                .map(|o| o.expect("all solve vars assigned"))
                .collect();
            (sol, h)
        })
        .collect();

    Ok(BacksolveOutcome::Finite(solutions))
}

/// Is this partial assignment already inconsistent with a generator all of
/// whose solve variables it binds?
///
/// Used only to tell "this branch is dead" from "this branch is
/// under-determined"; an undecidable answer is reported as `false`, which is
/// the conservative direction (the caller then declines rather than pruning).
fn partial_is_refuted(
    gens: &[GbPoly],
    partial: &[Option<ExprId>],
    vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> bool {
    let n_vars = vars.len();
    if n_solve != n_vars {
        // Parameters: a generator whose unknowns are all bound and which is not
        // the zero function of the parameters at this branch refutes it
        // generically (`x = 0` in `c·y − d·x·y + …` leaves `c·y`, …).
        let named: Vec<ExprId> = partial
            .iter()
            .zip(vars)
            .map(|(p, &v)| p.unwrap_or(v))
            .collect();
        return gens.iter().any(|g| {
            let bound = g.terms.keys().all(|e| {
                e.iter()
                    .take(n_solve)
                    .enumerate()
                    .all(|(i, &k)| k == 0 || partial[i].is_some())
            });
            bound
                && gbpoly_to_expr(g, &named, pool).is_some_and(|e| {
                    matches!(
                        generic::status(e, vars, n_solve, pool),
                        generic::Generic::Nonzero(_)
                    )
                })
        });
    }
    let mut evaluator = verify::CBallEval::default();
    let mut values: Vec<Option<verify::CBall>> = Vec::with_capacity(n_vars);
    for slot in partial.iter().take(n_vars) {
        values.push(match slot {
            Some(v) => evaluator.eval(*v, pool).ok(),
            None => None,
        });
    }
    gens.iter()
        .any(|g| verify::poly_residual_partial(g, &values).is_some_and(|r| r.excludes_zero()))
}

/// The solver's post-condition: drop every candidate that provably fails the
/// original system, and collapse candidates that cannot be told apart.
///
/// Substituting a finished tuple back into the equations costs a handful of
/// ball multiplications — orders of magnitude less than the Gröbner basis that
/// produced it — so it runs unconditionally rather than behind a flag.  The
/// test is one-sided by construction (see [`verify`]): a tuple is removed only
/// when its residual ball is *separated* from zero, so a genuine solution can
/// never be filtered out.
///
/// Tuples containing a free parameter are not numbers, and are checked as
/// functions of the parameters instead (see [`generic`]): a tuple that is
/// undefined (`0·0⁻¹`) or whose residual is provably not the zero function is
/// not a solution for generic parameter values and is dropped; one whose
/// residual cannot be decided either way is kept.
fn refine_solutions(
    solutions: Vec<Candidate>,
    orig_polys: &[GbPoly],
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Vec<Candidate> {
    let n_vars = all_vars.len();
    let mut kept: Vec<Candidate> = Vec::new();
    let mut kept_values: Vec<Vec<verify::CBall>> = Vec::new();
    let mut kept_parametric: Vec<Solution> = Vec::new();
    let mut evaluator = verify::CBallEval::default();

    for (sol, hyps) in solutions {
        if crate::budget::interrupted() {
            // The armed caller discards the result.
            break;
        }
        let mut values: Vec<verify::CBall> = Vec::with_capacity(n_vars);
        let mut gap = None;
        for &v in &sol {
            match evaluator.eval(v, pool) {
                Ok(b) => values.push(b),
                Err(g) => {
                    gap = Some(g);
                    break;
                }
            }
        }
        match gap {
            // A free parameter: decide the tuple as a function of them.
            Some(verify::VerifyGap::Unsupported) if n_solve < n_vars => {
                if !parametric_tuple_survives(&sol, orig_polys, all_vars, n_solve, pool) {
                    continue;
                }
                if kept_parametric
                    .iter()
                    .any(|prev| same_generic_point(prev, &sol, all_vars, n_solve, pool))
                {
                    continue;
                }
                kept_parametric.push(sol.clone());
                kept.push((sol, hyps));
                continue;
            }
            // A node the checker does not model: nothing can be proved, so
            // nothing is claimed — keep it as it was produced.
            Some(verify::VerifyGap::Unsupported) => {
                kept.push((sol, hyps));
                continue;
            }
            // `0/0`, `0^-1`: the tuple denotes no point of ℂⁿ.
            Some(verify::VerifyGap::Undefined) => continue,
            None => {}
        }
        // Free parameters occupy the tail of the indeterminate list; a tuple
        // that evaluated fully cannot have any, so `values` covers every
        // indeterminate the polynomials mention.
        if values.len() < n_vars {
            kept.push((sol, hyps));
            continue;
        }
        if verify::is_refuted(orig_polys, &values) {
            continue;
        }
        if kept_values
            .iter()
            .any(|prev| verify::same_point(prev, &values))
        {
            continue;
        }
        kept_values.push(values);
        kept.push((sol, hyps));
    }
    kept
}

/// Substitute a parametric candidate into the system: `false` when a
/// coordinate is undefined for every parameter value tried, or when a residual
/// is provably not the zero function of the parameters.
///
/// Sound in the dropping direction: a genuine solution of the ideal makes every
/// residual the zero function (each radical the back-substitution introduced
/// squares to its radicand identically), so a residual separated from zero at
/// even one parameter point, or with a non-zero cleared numerator, proves the
/// tuple is not a solution for generic parameters.
fn parametric_tuple_survives(
    sol: &Solution,
    orig_polys: &[GbPoly],
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> bool {
    if sol
        .iter()
        .any(|&v| generic::is_undefined(v, all_vars, n_solve, pool))
    {
        return false;
    }
    let mut named: Vec<ExprId> = sol.clone();
    named.extend_from_slice(&all_vars[n_solve..]);
    for p in orig_polys {
        if crate::budget::interrupted() {
            return true;
        }
        let Some(r) = gbpoly_to_expr(p, &named, pool) else {
            continue;
        };
        if matches!(
            generic::status(r, all_vars, n_solve, pool),
            generic::Generic::Nonzero(_)
        ) {
            return false;
        }
    }
    true
}

/// Are two parametric tuples provably the same point for every parameter
/// value?  Structural identity, or a coordinate-wise difference that is the
/// zero function.  `false` when undecided, so distinct solutions are never
/// merged.
fn same_generic_point(
    a: &Solution,
    b: &Solution,
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> bool {
    a.iter().zip(b).all(|(&u, &v)| {
        u == v
            || matches!(
                generic::status(
                    pool.add(vec![u, neg_expr(v, pool)]),
                    all_vars,
                    n_solve,
                    pool
                ),
                generic::Generic::Zero
            )
    })
}

// ---------------------------------------------------------------------------
// Poles: undoing the equivalence that clearing denominators broke
// ---------------------------------------------------------------------------

/// What could be established about a cleared denominator at a candidate root.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PoleStatus {
    /// Proved non-zero — the root survives with no hypothesis attached.
    Nonzero,
    /// Proved zero (or the denominator is not even defined there): the root is
    /// spurious and is removed.  This is a *positive* finding, so an empty
    /// solution list that follows from it is a real "no solution".
    Vanishes,
    /// A number, but one the enclosure could not separate from zero.  Treated
    /// as vanishing — the safe direction — but recorded, because unlike
    /// [`PoleStatus::Vanishes`] it is not a proof and must not be allowed to
    /// underwrite the claim "this system has no solutions".
    Indeterminate,
    /// Still mentions a free parameter, so it cannot be decided at all.  The
    /// root is kept under a stated non-vanishing hypothesis.
    Undecided,
}

/// Decide whether a domain condition, already specialised at a candidate root,
/// vanishes there.
///
/// Four sources are consulted in order of strength: exact rational arithmetic,
/// the structural zero test, the ambient assumptions (`R1 > 0` settles `R1 ≠ 0`
/// outright, which is why declaring a resistance positive removes a side
/// condition), and finally rigorous ball arithmetic.
fn domain_status(e: ExprId, pool: &ExprPool) -> PoleStatus {
    if let Some(v) = rational_value(e, pool) {
        return if v == 0 {
            PoleStatus::Vanishes
        } else {
            PoleStatus::Nonzero
        };
    }
    if is_certain_zero(e, pool) {
        return PoleStatus::Vanishes;
    }
    match crate::simplify::assumptions::assumed_sign(e, pool) {
        Some(crate::simplify::assumptions::Sign::Zero) => return PoleStatus::Vanishes,
        Some(_) => return PoleStatus::Nonzero,
        None => {}
    }
    match verify::CBallEval::default().eval(e, pool) {
        Ok(ball) if ball.excludes_zero() => PoleStatus::Nonzero,
        // A number whose enclosure straddles zero.  For a denominator that
        // really vanishes this is the *exact* answer arriving as a tight ball
        // around zero; for one that does not it would need to be smaller than
        // 2^-180, which no root of the systems this solver accepts is.
        Ok(_) => PoleStatus::Indeterminate,
        // `0^-1` inside the denominator itself: the root is outside the
        // equation's domain either way.
        Err(verify::VerifyGap::Undefined) => PoleStatus::Vanishes,
        Err(verify::VerifyGap::Unsupported) => PoleStatus::Undecided,
    }
}

/// Rebuild `expr` with each `vars[i]` replaced by `values[i]`.
///
/// Memoised on [`ExprId`], so a shared sub-expression is rewritten once: the
/// input is a DAG and its tree expansion is not what anyone wants to walk.
fn substitute_solution(
    expr: ExprId,
    vars: &[ExprId],
    values: &[ExprId],
    pool: &ExprPool,
    memo: &mut std::collections::HashMap<ExprId, ExprId>,
) -> ExprId {
    if let Some(&hit) = memo.get(&expr) {
        return hit;
    }
    if let Some(i) = vars.iter().position(|&v| v == expr) {
        memo.insert(expr, values[i]);
        return values[i];
    }
    let out = match pool.get(expr) {
        ExprData::Add(args) => {
            let new: Vec<ExprId> = args
                .iter()
                .map(|&a| substitute_solution(a, vars, values, pool, memo))
                .collect();
            pool.add(new)
        }
        ExprData::Mul(args) => {
            let new: Vec<ExprId> = args
                .iter()
                .map(|&a| substitute_solution(a, vars, values, pool, memo))
                .collect();
            pool.mul(new)
        }
        ExprData::Pow { base, exp } => {
            let b = substitute_solution(base, vars, values, pool, memo);
            let e = substitute_solution(exp, vars, values, pool, memo);
            pool.pow(b, e)
        }
        ExprData::Func { name, args } => {
            let new: Vec<ExprId> = args
                .iter()
                .map(|&a| substitute_solution(a, vars, values, pool, memo))
                .collect();
            pool.func(&name, new)
        }
        _ => expr,
    };
    memo.insert(expr, out);
    out
}

/// The verdict of substituting a candidate back into the caller's own equations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OriginalCheck {
    /// Nothing could be held against it.
    Survives,
    /// The residual is *separated* from zero: a proof that this is not a root.
    Refuted,
    /// The expression has no value at the candidate (`1/0`, `0/0`) — or its
    /// reciprocal's argument merely could not be separated from zero, which
    /// reads the same from here.  A drop either way, but not a proof.
    Undefined,
}

/// Substitute a candidate into the **original** rational equations — the ones
/// the caller wrote, before anything was multiplied through.
///
/// Independent of the denominator test above and deliberately so: this walks
/// the caller's own expression, so a mistake in clearing denominators shows up
/// here rather than being confirmed by its own output.  One-sided in the same
/// direction as [`refine_solutions`] — a candidate is only ever dropped, never
/// rescued, and a residual ball that straddles zero proves nothing and lets the
/// candidate through.
fn check_against_original(
    sol: &Solution,
    equations: &[ExprId],
    vars: &[ExprId],
    pool: &ExprPool,
) -> OriginalCheck {
    let mut memo = std::collections::HashMap::new();
    let mut evaluator = verify::CBallEval::default();
    let mut verdict = OriginalCheck::Survives;
    for &eq in equations {
        let at_root = substitute_solution(eq, vars, sol, pool, &mut memo);
        match evaluator.eval(at_root, pool) {
            Ok(ball) if ball.excludes_zero() => return OriginalCheck::Refuted,
            Ok(_) => {}
            Err(verify::VerifyGap::Undefined) => verdict = OriginalCheck::Undefined,
            // A free parameter: nothing can be concluded, and nothing is.
            Err(verify::VerifyGap::Unsupported) => {}
        }
    }
    verdict
}

/// Remove the roots that clearing denominators invented, and state the
/// hypotheses under which the rest stand.
///
/// `N/D = 0` is equivalent to `N = 0 ∧ D ≠ 0`; the solver was handed only
/// `N = 0`, so every root it produced has to be re-tested against the
/// `domains` polynomial that says where its equation has a value at all
/// (see [`rational`] — that is *not* the same thing as the product of the
/// denominators). `x/(x−1) = 1/(x−1)` clears to `(x−1)² = 0` and its only root
/// `x = 1` is removed here, which is the whole reason this step exists:
/// returning it would be a wrong answer.
///
/// Three outcomes, all visible to the caller:
///
/// * **Proved non-zero** — the root is returned unconditionally.
/// * **Proved zero** — the root is dropped.  Because this is a proof, an empty
///   result is a genuine "no solution".
/// * **Undecidable** (the condition still mentions a free parameter, so whether
///   it vanishes depends on values nobody supplied) — the root is returned
///   under a `≠ 0` hypothesis recorded through [`assume_nonzero`] and reported
///   by [`take_solve_side_conditions`].  It is never assumed silently.
///
/// Returns the surviving roots and whether any drop was
/// [`PoleStatus::Indeterminate`] rather than proved; the caller must not
/// report "no solutions" on the strength of one of those.
fn exclude_pole_roots(
    solutions: Vec<Candidate>,
    domains: &[GbPoly],
    equations: &[ExprId],
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> (Vec<Candidate>, bool) {
    if domains.is_empty() {
        return (solutions, false);
    }
    let solve_vars: Vec<ExprId> = all_vars[..n_solve].to_vec();
    let mut kept = Vec::with_capacity(solutions.len());
    let mut unproved_drop = false;

    for (sol, mut branch_hyps) in solutions {
        // Name exponent slot `i` by the solved value for an unknown, and by
        // the parameter itself for everything past them.
        let mut at_root: Vec<ExprId> = sol.clone();
        at_root.extend_from_slice(&all_vars[n_solve..]);

        let mut hypotheses: Vec<ExprId> = Vec::new();
        let mut excluded = false;
        for d in domains {
            let Some(value) = gbpoly_to_expr(d, &at_root, pool) else {
                // `at_root` names every indeterminate by construction, so this
                // is unreachable; refusing the root is the safe reading.
                excluded = true;
                unproved_drop = true;
                break;
            };
            match domain_status(value, pool) {
                PoleStatus::Nonzero => {}
                PoleStatus::Vanishes => {
                    excluded = true;
                    break;
                }
                PoleStatus::Indeterminate => {
                    excluded = true;
                    unproved_drop = true;
                    break;
                }
                PoleStatus::Undecided => hypotheses.push(value),
            }
        }
        if excluded {
            continue;
        }
        match check_against_original(&sol, equations, &solve_vars, pool) {
            OriginalCheck::Survives => {}
            OriginalCheck::Refuted => continue,
            OriginalCheck::Undefined => {
                // The product-of-denominators test above did not object, so the
                // two enclosures disagree.  Dropping is the safe reading, but
                // it is not the proof `Vanishes` would have been.
                unproved_drop = true;
                continue;
            }
        }
        for h in hypotheses {
            if !branch_hyps.contains(&h) {
                branch_hyps.push(h);
            }
        }
        kept.push((sol, branch_hyps));
    }
    (kept, unproved_drop)
}

/// Free symbols in `equations` that are not among the declared solve `vars`,
/// in stable [`ExprId`] order (via [`collect_free_vars`]'s `BTreeSet`).
///
/// [`solve_polynomial_system`] appends these after `vars`, and the resulting
/// concatenation is the exponent-vector ordering of any
/// [`SolutionSet::Parametric`] basis it returns — so a caller that wants to
/// read that basis back with [`gbpoly_to_expr`] needs this list.
pub fn collect_parameters(equations: &[ExprId], vars: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let declared: BTreeSet<ExprId> = vars.iter().copied().collect();
    let mut params = BTreeSet::new();
    for &eq in equations {
        for v in collect_free_vars(eq, pool) {
            if !declared.contains(&v) {
                params.insert(v);
            }
        }
    }
    params.into_iter().collect()
}

/// Is the basis `⟨1⟩`?
fn is_unit_ideal(gens: &[GbPoly], n_vars: usize) -> bool {
    gens.len() == 1
        && gens[0].terms.len() == 1
        && gens[0].leading_exp(MonomialOrder::Lex) == Some(vec![0u32; n_vars])
}

/// A non-zero basis element in the parameters alone, if there is one.  Under a
/// Lex order with the parameters last this is exactly the elimination ideal
/// `I ∩ ℚ[params]` being non-trivial.
fn parameter_condition(gens: &[GbPoly], n_solve: usize) -> Option<&GbPoly> {
    gens.iter().find(|g| {
        !g.is_zero()
            && g.terms
                .keys()
                .all(|e| e.iter().take(n_solve).all(|&k| k == 0))
    })
}

/// Most factor-choice subsystems [`solve_by_factor_branches`] will enumerate.
const MAX_FACTOR_BRANCHES: usize = 64;

/// Solve a parametric system as the union over every choice of one irreducible
/// factor per equation.
///
/// `Ok(None)` means "not applicable or not conclusive" — nothing factors, the
/// product of choices is too large, or some subsystem has no complete
/// triangular elimination (a positive-dimensional component) — and the caller
/// solves the whole system instead.  A subsystem that needs a univariate of
/// degree > 2 refuses for the whole system (`E-SOLVE-002`).  A subsystem whose basis
/// contains a non-zero polynomial in the parameters alone is generically empty
/// and contributes nothing.  The candidates returned are the union, unfiltered:
/// the caller substitutes them into the original equations.
fn solve_by_factor_branches(
    polys: &[GbPoly],
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> Result<Option<Vec<Candidate>>, SolverError> {
    let n_vars = all_vars.len();
    let mut choices: Vec<Vec<GbPoly>> = Vec::with_capacity(polys.len());
    let mut product = 1usize;
    // A factor in the parameters alone (`a` in `a·x·(1 − y)`) is dropped from
    // every branch, which assumes it does not vanish; that is stated.
    let mut dropped: Vec<ExprId> = Vec::new();
    for p in polys {
        let Some((factors, params_only)) = generic::split_factors(p, all_vars, n_solve) else {
            return Ok(None);
        };
        if factors.is_empty() {
            return Ok(None);
        }
        for f in &params_only {
            for h in generic::nonzero_hypotheses(f, all_vars, pool) {
                if !dropped.contains(&h) {
                    dropped.push(h);
                }
            }
        }
        product = product.saturating_mul(factors.len());
        choices.push(factors);
    }
    if product <= 1 || product > MAX_FACTOR_BRANCHES {
        return Ok(None);
    }
    let mut candidates = Vec::new();
    for combo in 0..product {
        if crate::budget::interrupted() {
            return Err(interrupted_error());
        }
        let mut idx = combo;
        let mut sub = Vec::with_capacity(choices.len());
        for fs in &choices {
            sub.push(fs[idx % fs.len()].clone());
            idx /= fs.len();
        }
        let gb = GroebnerBasis::compute(sub, MonomialOrder::Lex);
        if crate::budget::interrupt_pending() {
            return Err(interrupted_error());
        }
        let gens = gb.generators();
        if is_unit_ideal(gens, n_vars) || parameter_condition(gens, n_solve).is_some() {
            continue;
        }
        match try_backsolve_generators(gens, all_vars, n_solve, pool) {
            Ok(BacksolveOutcome::Finite(c)) => {
                candidates.extend(c.into_iter().map(|(sol, mut hyps)| {
                    for &h in &dropped {
                        if !hyps.contains(&h) {
                            hyps.push(h);
                        }
                    }
                    (sol, hyps)
                }))
            }
            Ok(BacksolveOutcome::NoSolution) => {}
            Ok(BacksolveOutcome::Stuck) => return Ok(None),
            // A component of the variety needs an irreducible univariate of
            // degree > 2; so does the whole system, whose basis is far costlier
            // to compute only to refuse the same way.
            Err(e) => return Err(e),
        }
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    Ok(Some(candidates))
}

/// Solve a polynomial — or **rational** — system in the declared unknowns.
///
/// `equations` — list of `ExprId` each representing `p = 0`.
/// `vars` — unknowns to solve for (order used for `GbPoly` exponent vectors).
///
/// Symbols that appear in `equations` but are absent from `vars` are treated as
/// free parameters: solutions may be expressions in those symbols (e.g.
/// `x² − y = 0` in `[x]` yields `x = ±√y`).
///
/// # Rational equations
///
/// An equation may be a ratio of polynomials — `(Vo − Vin)/R1 + Vo·s·C`, the
/// shape every admittance equation in nodal analysis has.  Each is put over a
/// common denominator and the numerator system is solved.
///
/// That step is **not** an equivalence: `N/D = 0` means `N = 0 ∧ D ≠ 0`, and a
/// root of `N` at which `D` vanishes is not a solution.  Such roots are removed
/// again by `exclude_pole_roots`, which decides `D ≠ 0` exactly where it can
/// (rational arithmetic, the ambient assumptions, ball arithmetic) and
/// otherwise records a `D ≠ 0` hypothesis through [`take_solve_side_conditions`]
/// rather than assuming it.  Surviving roots are additionally substituted into
/// the equations as the caller wrote them, before anything was cleared.
///
/// Returns a [`SolutionSet`] with symbolic `ExprId` values for each solution
/// (parallel to `vars` only — parameters are not included in solution tuples).
///
/// # Post-condition
///
/// Every parameter-free tuple in a [`SolutionSet::Finite`] has been substituted
/// back into the **input** equations and survived: its residual could not be
/// separated from zero in rigorous ball arithmetic.  Checking costs a few
/// hundred microseconds against a Gröbner basis that is superexponential in the
/// worst case, so it is unconditional rather than opt-in.  A tuple whose
/// coordinates are not numbers (a `0/0` produced by a degenerate division) is
/// dropped for the same reason, and tuples that denote the same point are
/// reported once.
///
/// # Hypotheses
///
/// A *parametric* tuple is not a number, so it is checked as a function of the
/// parameters instead: substituted into the input equations, every residual
/// must be the zero function (decided exactly for rational functions of the
/// parameters, by ball arithmetic at sample parameter points otherwise), and a
/// tuple that is undefined or provably not a solution is dropped.  It holds
/// under the non-vanishing assumptions its branch made about leading
/// coefficients that mention free parameters, and those are not left unsaid —
/// see [`take_solve_side_conditions`], which must be read before the next call
/// on this thread.  A non-zero polynomial in the parameters alone in the ideal
/// gives [`SolutionSet::NoSolution`] (no solution for generic parameters, the
/// polynomial's factors stated non-zero).
///
/// # Declared domains
///
/// An unknown declared `Positive`, `NonNegative`, `NonZero` or `Integer` gets
/// no value provably outside that domain; an undecided membership is kept and
/// reported as a side condition.
///
/// # Budget
///
/// The active [`crate::budget`] is polled in the Gröbner basis computation and
/// in back-substitution. A trip returns `Err(SolverError::NotPolynomial(..))`
/// whose message names the trip, never a partial solution set; the
/// [`crate::budget::BudgetTrip`] itself is left in [`crate::budget::take_trip`]
/// (`SolverError` is exhaustive, so it cannot carry a budget variant), which
/// is how the Python binding raises `BudgetExceededError` instead.
pub fn solve_polynomial_system(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<SolutionSet, SolverError> {
    let interrupts = crate::budget::arm();
    let result = solve_polynomial_system_unguarded(equations, vars.clone(), pool);
    match interrupts.trip() {
        // Whatever came back was computed from a basis that stopped part-way.
        Some(trip) => Err(budget_stopped(trip)),
        None => match result {
            Ok(SolutionSet::Finite(sols)) => Ok(SolutionSet::Finite(filter_by_declared_domain(
                sols, &vars, pool,
            ))),
            other => other,
        },
    }
}

/// Is a solved value a member of the domain its unknown was declared with?
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Membership {
    Yes,
    No,
    Unknown,
}

/// Decide `value ∈ domain` for the domains [`filter_by_declared_domain`]
/// enforces: exactly for a rational, by a rigorous enclosure for a constant,
/// and through the ambient sign assumptions for a parametric value.
fn domain_membership(value: ExprId, domain: crate::kernel::Domain, pool: &ExprPool) -> Membership {
    use crate::kernel::Domain;
    use crate::simplify::assumptions::{assumed_sign, Sign};
    let yes_if = |b: bool| if b { Membership::Yes } else { Membership::No };
    // The back-substitution leaves `(0 + √16)/2` unfolded; the exact value
    // decides membership outright where an enclosure can only bracket it.
    let exact =
        rational_value(value, pool).or_else(|| rational_value(simplified(value, pool), pool));
    if let Some(q) = exact {
        return match domain {
            Domain::Positive => yes_if(q > 0),
            Domain::NonNegative => yes_if(q >= 0),
            Domain::NonZero => yes_if(q != 0),
            Domain::Integer => yes_if(*q.denom() == 1),
            Domain::Real | Domain::Complex => Membership::Yes,
        };
    }
    match verify::CBallEval::default().eval(value, pool) {
        Ok(b) => {
            let re = b.re_ball();
            let im = b.im_ball();
            let real = im.is_exact() && im.mid_f64() == 0.0;
            let non_real = !im.contains(0.0);
            match domain {
                Domain::NonZero if b.excludes_zero() => Membership::Yes,
                Domain::NonZero => Membership::Unknown,
                // Each of these implies a real value.
                _ if non_real => Membership::No,
                Domain::Positive if re.hi() <= 0 => Membership::No,
                Domain::Positive if re.lo() > 0 && real => Membership::Yes,
                Domain::NonNegative if re.hi() < 0 => Membership::No,
                Domain::NonNegative if re.lo() >= 0 && real => Membership::Yes,
                Domain::Integer => {
                    let (lo, hi) = (re.lo(), re.hi());
                    let floor_lo = lo.clone().floor();
                    if lo.is_finite() && floor_lo == hi.clone().floor() && lo > floor_lo {
                        Membership::No
                    } else {
                        Membership::Unknown
                    }
                }
                _ => Membership::Unknown,
            }
        }
        Err(_) => match (domain, assumed_sign(value, pool)) {
            (Domain::Positive, Some(Sign::Positive)) => Membership::Yes,
            (Domain::Positive, Some(_)) => Membership::No,
            (Domain::NonNegative, Some(Sign::Positive | Sign::Zero)) => Membership::Yes,
            (Domain::NonNegative, Some(Sign::Negative)) => Membership::No,
            (Domain::NonZero, Some(Sign::Positive | Sign::Negative)) => Membership::Yes,
            (Domain::NonZero, Some(Sign::Zero)) => Membership::No,
            _ => Membership::Unknown,
        },
    }
}

/// `value` through the simplifier, for an exact reading and a legible side
/// condition.
fn simplified(value: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::engine::simplify(value, pool).value
}

/// Honour the domain each unknown was **declared** with.
///
/// `pool.symbol("x", Domain::Positive)` says `x > 0`; `solve([x + 3], [x])`
/// returning `x = −3` contradicts the caller's own declaration, the same way
/// `simplify` would be wrong to ignore it.  A value provably outside the
/// declared domain is dropped; one that cannot be decided (it depends on a
/// parameter of unknown sign) is kept and the membership is stated as a side
/// condition rather than assumed.
///
/// `Positive`, `NonNegative`, `NonZero` and `Integer` are enforced.  `Real` is
/// not: it is the default domain of every symbol, so it does not distinguish a
/// declaration from an omission, and `solve` documents its complex roots as the
/// default (`domain="real"` is the explicit filter).
fn filter_by_declared_domain(
    sols: Vec<Solution>,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Vec<Solution> {
    use crate::kernel::Domain;
    let declared: Vec<Option<Domain>> = vars
        .iter()
        .map(|&v| match pool.get(v) {
            ExprData::Symbol { domain, .. } => match domain {
                Domain::Positive | Domain::NonNegative | Domain::NonZero | Domain::Integer => {
                    Some(domain)
                }
                Domain::Real | Domain::Complex => None,
            },
            _ => None,
        })
        .collect();
    if declared.iter().all(Option::is_none) {
        return sols;
    }
    let mut out = Vec::with_capacity(sols.len());
    'sol: for sol in sols {
        let mut conds = Vec::new();
        for (&value, d) in sol.iter().zip(&declared) {
            let Some(d) = *d else { continue };
            match domain_membership(value, d, pool) {
                Membership::Yes => {}
                Membership::No => continue 'sol,
                Membership::Unknown => {
                    let shown = simplified(value, pool);
                    conds.push(match d {
                        Domain::Positive => crate::deriv::log::SideCondition::Positive(shown),
                        _ => crate::deriv::log::SideCondition::InDomain(shown, d),
                    })
                }
            }
        }
        for c in conds {
            record_side_condition(c);
        }
        out.push(sol);
    }
    out
}

/// The error [`solve_polynomial_system`] reports for a budget trip.
pub(crate) fn budget_stopped(trip: crate::budget::BudgetTrip) -> SolverError {
    crate::budget::record_trip(trip);
    SolverError::NotPolynomial(format!(
        "stopped by the active budget before the system was solved — this is not a statement \
         about the system: {trip} [{}]",
        trip.code()
    ))
}

/// A placeholder error for leaving the solver early once a budget trip is
/// recorded; [`solve_polynomial_system`] replaces it with [`budget_stopped`].
fn interrupted_error() -> SolverError {
    SolverError::NotPolynomial("stopped by the active budget".into())
}

fn solve_polynomial_system_unguarded(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<SolutionSet, SolverError> {
    // Hypotheses describe *this* call; a caller reading them after it must
    // never see one left behind by an earlier solve.  Same for the refusal
    // channel: a stale one would re-label an unrelated `NotPolynomial`.
    let _ = take_solve_side_conditions();
    forget_undefined_equation();
    let n_solve = vars.len();
    let vars_len = n_solve;
    let params = collect_parameters(&equations, &vars, pool);
    let mut all_vars = vars;
    all_vars.extend(params);
    let n_vars = all_vars.len();

    // Each equation is put over a common denominator; the numerators are the
    // system the Gröbner machinery below actually sees, and `domains` carries
    // what has to be non-zero for each numerator to speak for its equation.
    // For polynomial input the domain is `1` and nothing about this call
    // changes.
    let mut polys: Vec<GbPoly> = Vec::with_capacity(equations.len());
    let mut domains: Vec<GbPoly> = Vec::new();
    for eq in &equations {
        let cleared = match rational::clear_denominators(*eq, &all_vars, pool) {
            Ok(c) => c,
            Err(SolverError::NotPolynomial(msg)) if !undefined_equation_pending() => {
                // A coefficient that is not polynomial — `sin(q₁)` multiplying
                // `q̈₁` — is still a coefficient. Retry with every such
                // sub-expression free of the unknowns frozen into a parameter;
                // that path checks its own answers (see `opaque`).
                if let Some(out) =
                    opaque::solve_with_opaque_coefficients(&equations, &all_vars[..vars_len], pool)
                {
                    return out;
                }
                // An unknown that occurs only inside one invertible function
                // (`exp(x) + y = 3`) is solved for through that function.
                if let Some(out) = transcendental::solve_with_invertible_kernels(
                    &equations,
                    &all_vars[..vars_len],
                    pool,
                ) {
                    return out;
                }
                return Err(SolverError::NotPolynomial(msg));
            }
            Err(e) => return Err(e),
        };
        if !cleared.is_polynomial() {
            domains.push(cleared.domain);
        }
        polys.push(cleared.numer);
    }

    let has_params = n_solve < n_vars;

    // Candidates are checked against the *input* equations rather than the
    // basis: that is the contract the caller stated, and it does not inherit
    // any mistake the basis computation might have made.
    let finish = |candidates: Vec<Candidate>| -> Option<SolutionSet> {
        let had_candidates = !candidates.is_empty();
        let refined = refine_solutions(candidates, &polys, &all_vars, n_solve, pool);
        if had_candidates && refined.is_empty() {
            // Over ℂ a proper ideal always has a zero, so a candidate set that
            // is entirely refuted means the enumeration itself was unsound.
            // Reporting `Finite([])` here would be the worst possible answer —
            // "this system has no solutions" — so decline instead.
            return None;
        }
        // Clearing denominators enlarged the solution set; take the excess
        // back out.  An empty list *is* the right answer here — `1/(x−1) =
        // x/(x−1)` genuinely has none — but only when every drop was proved,
        // so a merely-undecided one falls through to the basis instead.
        let (kept, unproved_drop) =
            exclude_pole_roots(refined, &domains, &equations, &all_vars, n_solve, pool);
        if kept.is_empty() && unproved_drop {
            return None;
        }
        // Only the branches that survived contribute their hypotheses.
        let mut out = Vec::with_capacity(kept.len());
        for (sol, hyps) in kept {
            for h in hyps {
                assume_nonzero(h);
            }
            out.push(sol);
        }
        Some(SolutionSet::Finite(out))
    };

    // A parametric equation that factors is the union of its factors' zero
    // sets, and each choice of one factor per equation is a smaller system
    // whose leading coefficients are far less likely to vanish on a branch.
    // This is how equilibrium systems are written (`x·(a − b·y)`), and it is
    // what keeps `x = 0` and `y = a/b` from being produced by the same
    // division.  It runs before the basis of the whole system, whose Lex
    // basis over ℚ[unknowns, params] is far larger than its factors' are.
    if has_params {
        if let Some(candidates) = solve_by_factor_branches(&polys, &all_vars, n_solve, pool)? {
            if let Some(set) = finish(candidates) {
                return Ok(set);
            }
        }
    }

    let gb = GroebnerBasis::compute(polys.clone(), MonomialOrder::Lex);
    if crate::budget::interrupt_pending() {
        return Err(interrupted_error());
    }
    let gens = gb.generators();

    // Trivial ideal ⟨1⟩ → no solution.
    if is_unit_ideal(gens, n_vars) {
        return Ok(SolutionSet::NoSolution);
    }

    // A non-zero polynomial in the parameters alone lies in the ideal: the
    // system has a solution only where it vanishes, i.e. for no generic
    // parameter value (`[x − a, x − b]` needs `a = b`).  Back-substitution
    // never looks at such a generator — it mentions no unknown — so before this
    // check it returned the branch as though the condition held.
    if has_params {
        if let Some(cond) = parameter_condition(gens, n_solve) {
            for h in generic::nonzero_hypotheses(cond, &all_vars, pool) {
                assume_nonzero(h);
            }
            return Ok(SolutionSet::NoSolution);
        }
    }

    match try_backsolve_generators(gens, &all_vars, n_solve, pool)? {
        BacksolveOutcome::Finite(solutions) => {
            if let Some(set) = finish(solutions) {
                return Ok(set);
            }
        }
        BacksolveOutcome::NoSolution => return Ok(SolutionSet::NoSolution),
        BacksolveOutcome::Stuck => {}
    }

    // The full basis had no complete triangular elimination (or its candidates
    // did not survive): retry from a regular chain extracted from the same
    // basis.  A regular chain lies in the ideal, so its solution set contains
    // the true one and the post-condition filter still applies.
    let chain = extract_regular_chain_from_basis(gens, n_vars, MonomialOrder::Lex);
    if !chain.polys.is_empty() {
        if let BacksolveOutcome::Finite(solutions) =
            try_backsolve_generators(&chain.polys, &all_vars, n_solve, pool)?
        {
            if let Some(set) = finish(solutions) {
                return Ok(set);
            }
        }
    }
    // The basis being returned is the *numerator* ideal, whose variety contains
    // points the original equations are not even defined at.  Nobody filtered
    // them, because no solution list was produced to filter; say so rather than
    // hand back a basis that quietly means something wider than it looks.
    for d in &domains {
        if let Some(e) = gbpoly_to_expr(d, &all_vars, pool) {
            assume_nonzero(e);
        }
    }
    Ok(SolutionSet::Parametric(gb))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::eval_interp;
    use crate::kernel::{Domain, ExprPool};
    use std::collections::HashMap;

    fn eval_no_env(e: ExprId, pool: &ExprPool) -> f64 {
        eval_interp(e, &HashMap::new(), pool).expect("numeric eval")
    }

    fn has_numeric_pair(sols: &[Solution], pool: &ExprPool, expected: &[(f64, f64)]) -> bool {
        let tol = 1e-10;
        expected.iter().all(|(ex, ey)| {
            sols.iter().any(|s| {
                let x = eval_no_env(s[0], pool);
                let y = eval_no_env(s[1], pool);
                (x - ex).abs() < tol && (y - ey).abs() < tol
            })
        })
    }

    /// `expr_to_gbpoly` ∘ `gbpoly_to_expr` is the identity on the canonical
    /// side: an `Expr` rebuilt from a polynomial converts back to the same
    /// polynomial.
    #[test]
    fn gbpoly_expr_round_trip() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let vars = vec![x, y];

        // 3/2·x²y − y + 7
        let expr = pool.add(vec![
            pool.mul(vec![
                pool.rational(3_i32, 2_i32),
                pool.pow(x, pool.integer(2_i32)),
                y,
            ]),
            pool.mul(vec![pool.integer(-1_i32), y]),
            pool.integer(7_i32),
        ]);

        let p = expr_to_gbpoly(expr, &vars, &pool).unwrap();
        let back = gbpoly_to_expr(&p, &vars, &pool).expect("named every variable");
        let p2 = expr_to_gbpoly(back, &vars, &pool).unwrap();

        assert_eq!(p.n_vars, p2.n_vars);
        assert_eq!(p.terms, p2.terms);
    }

    #[test]
    fn gbpoly_to_expr_zero_and_constant() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let zero = GbPoly::zero(1);
        assert_eq!(
            gbpoly_to_expr(&zero, &[x], &pool),
            Some(pool.integer(0_i32))
        );

        let five = GbPoly::constant(Rational::from(5), 1);
        assert_eq!(
            gbpoly_to_expr(&five, &[x], &pool),
            Some(pool.integer(5_i32))
        );
    }

    /// A short `vars` list must refuse rather than silently rename exponent
    /// slots — a wrong-but-plausible polynomial is the worst outcome here.
    #[test]
    fn gbpoly_to_expr_refuses_a_short_variable_list() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);

        let p = expr_to_gbpoly(pool.mul(vec![x, y]), &[x, y], &pool).unwrap();

        assert_eq!(gbpoly_to_expr(&p, &[x], &pool), None);
        assert!(gbpoly_to_expr(&p, &[x, y], &pool).is_some());
    }

    /// The Gröbner basis of an ideal must survive being read out as `Expr` and
    /// fed back in — otherwise elimination results cannot be reused.
    #[test]
    fn basis_generators_round_trip_through_expr() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let vars = vec![x, y];
        let neg_one = pool.integer(-1_i32);

        // x² + y² − 1, x − y
        let circle = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.pow(y, pool.integer(2_i32)),
            neg_one,
        ]);
        let line = pool.add(vec![x, pool.mul(vec![neg_one, y])]);

        let gens = vec![
            expr_to_gbpoly(circle, &vars, &pool).unwrap(),
            expr_to_gbpoly(line, &vars, &pool).unwrap(),
        ];
        let gb = GroebnerBasis::compute_lex(gens);
        assert_eq!(gb.order(), MonomialOrder::Lex);

        for g in gb.generators() {
            let e = gbpoly_to_expr(g, &vars, &pool).expect("named every variable");
            let reparsed = expr_to_gbpoly(e, &vars, &pool).unwrap();
            assert!(gb.contains(&reparsed), "generator left the ideal");
        }
    }

    #[test]
    fn linear_system() {
        // x + y - 1 = 0, x - y = 0  →  x = 1/2, y = 1/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = pool.add(vec![x, y, neg_one]);
        let eq2 = pool.add(vec![x, pool.mul(vec![neg_one, y])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert!(has_numeric_pair(&sols, &pool, &[(0.5, 0.5)]));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn univariate_quadratic() {
        // x² - 1 = 0  →  x = ±1
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let eq = pool.add(vec![x2, neg_one]);
        let result = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            let vals: Vec<f64> = sols.iter().map(|s| eval_no_env(s[0], &pool)).collect();
            assert!(vals.iter().any(|v| (v - 1.0).abs() < 1e-10));
            assert!(vals.iter().any(|v| (v + 1.0).abs() < 1e-10));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn circle_line_intersection() {
        // x² + y² - 1 = 0,  y - x = 0  →  x = y = ±√2/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let two = pool.integer(2_i32);
        let x2 = pool.pow(x, two);
        let y2 = pool.pow(y, two);
        // x² + y² - 1
        let eq1 = pool.add(vec![x2, y2, neg_one]);
        // y - x
        let eq2 = pool.add(vec![y, pool.mul(vec![neg_one, x])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert_eq!(
                sols.len(),
                2,
                "expected exactly 2 solutions, got {}",
                sols.len()
            );
            let root = (0.5_f64).sqrt(); // √2/2
            assert!(has_numeric_pair(
                &sols,
                &pool,
                &[(root, root), (-root, -root)]
            ));
        } else {
            panic!("expected finite solution set");
        }
    }

    #[test]
    fn no_solution_inconsistent() {
        // x = 0 and x = 1 simultaneously → ⟨1⟩ ideal
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = x; // x = 0
        let eq2 = pool.add(vec![x, neg_one]); // x - 1 = 0
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x], &pool).unwrap();
        assert!(matches!(result, SolutionSet::NoSolution));
    }

    #[test]
    fn parabola_and_line() {
        // y - x² = 0,  y - x = 0  →  x(x-1)=0 → (0,0) and (1,1)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let two = pool.integer(2_i32);
        let x2 = pool.pow(x, two);
        // y - x²
        let eq1 = pool.add(vec![y, pool.mul(vec![neg_one, x2])]);
        // y - x
        let eq2 = pool.add(vec![y, pool.mul(vec![neg_one, x])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        if let SolutionSet::Finite(sols) = result {
            assert_eq!(sols.len(), 2);
            assert!(has_numeric_pair(&sols, &pool, &[(0.0, 0.0), (1.0, 1.0)]));
        } else {
            panic!("expected finite solution set");
        }
    }

    /// `x^k` as an `ExprId`.
    fn powk(pool: &ExprPool, base: ExprId, k: i32) -> ExprId {
        pool.pow(base, pool.integer(k))
    }

    fn finite(eqs: Vec<ExprId>, vars: Vec<ExprId>, pool: &ExprPool) -> Vec<Solution> {
        match solve_polynomial_system(eqs, vars, pool).expect("solve") {
            SolutionSet::Finite(s) => s,
            other => panic!(
                "expected a finite solution set, got {}",
                match other {
                    SolutionSet::NoSolution => "NoSolution",
                    _ => "Parametric",
                }
            ),
        }
    }

    #[test]
    fn spurious_tuple_is_refuted() {
        // x² − xy = 0, xy − y = 0.  y(x−1) = 0 forces y = 0 or x = 1;
        // y = 0 ⇒ x² = 0 ⇒ x = 0, and x = 1 ⇒ 1 − y = 0 ⇒ y = 1.
        // The solution set is exactly {(0,0), (1,1)}.  The tuple (−1, 1) has
        // residual x² − xy = 1 + 1 = 2 and must never be reported.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = pool.add(vec![powk(&pool, x, 2), pool.mul(vec![neg_one, x, y])]);
        let eq2 = pool.add(vec![pool.mul(vec![x, y]), pool.mul(vec![neg_one, y])]);
        let sols = finite(vec![eq1, eq2], vec![x, y], &pool);
        assert!(has_numeric_pair(&sols, &pool, &[(0.0, 0.0), (1.0, 1.0)]));
        assert_eq!(sols.len(), 2, "exactly two points, got {sols:?}");
    }

    #[test]
    fn vanishing_leading_coefficient_branch_is_kept() {
        // −3x − 2xy = −x(3 + 2y) = 0 and −3y − x² = 0.
        // y = −3/2 kills the first equation for every x, and then
        // x² = −3y = 9/2, so (±3/√2, −3/2) are solutions; (0,0) is the third.
        // The old back-solver divided by the leading coefficient 2y + 3, which
        // is zero on exactly that branch, and lost both of them.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.add(vec![
            pool.mul(vec![pool.integer(-3_i32), x]),
            pool.mul(vec![pool.integer(-2_i32), x, y]),
        ]);
        let eq2 = pool.add(vec![
            pool.mul(vec![pool.integer(-3_i32), y]),
            pool.mul(vec![pool.integer(-1_i32), powk(&pool, x, 2)]),
        ]);
        let sols = finite(vec![eq1, eq2], vec![x, y], &pool);
        let r = (4.5_f64).sqrt();
        assert!(has_numeric_pair(
            &sols,
            &pool,
            &[(0.0, 0.0), (r, -1.5), (-r, -1.5)]
        ));
        assert_eq!(sols.len(), 3, "exactly three points, got {sols:?}");
    }

    #[test]
    fn undefined_coordinate_is_not_a_solution() {
        // xy − y = 0, y − 2x² = 0 → {(0,0), (1,2)}.  The old back-solver
        // reported `0·0⁻¹` for the first coordinate, which is not a number.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.add(vec![
            pool.mul(vec![x, y]),
            pool.mul(vec![pool.integer(-1_i32), y]),
        ]);
        let eq2 = pool.add(vec![
            y,
            pool.mul(vec![pool.integer(-2_i32), powk(&pool, x, 2)]),
        ]);
        let sols = finite(vec![eq1, eq2], vec![x, y], &pool);
        assert!(has_numeric_pair(&sols, &pool, &[(0.0, 0.0), (1.0, 2.0)]));
        assert_eq!(sols.len(), 2, "exactly two points, got {sols:?}");
    }

    #[test]
    fn unfolded_vanishing_discriminant_is_recognised() {
        // The pool folds no arithmetic on literals: `0^2` and `0 * 4 * 1` both
        // survive as nodes, so a purely structural zero test misses the
        // discriminant of x² = 0 and of (x−1)² = 0 alike.
        let pool = ExprPool::new();
        let zero = pool.integer(0_i32);
        let b2 = pool.pow(zero, pool.integer(2_i32));
        let four_ac = pool.mul(vec![pool.integer(4_i32), pool.integer(1_i32), zero]);
        let disc = pool.add(vec![b2, pool.mul(vec![pool.integer(-1_i32), four_ac])]);
        assert!(is_zero_value(disc, &pool), "0² − 4·1·0 = 0");

        let b2 = pool.pow(pool.integer(-2_i32), pool.integer(2_i32));
        let four_ac = pool.mul(vec![
            pool.integer(4_i32),
            pool.integer(1_i32),
            pool.integer(1_i32),
        ]);
        let disc = pool.add(vec![b2, pool.mul(vec![pool.integer(-1_i32), four_ac])]);
        assert!(is_zero_value(disc, &pool), "(−2)² − 4·1·1 = 0");

        // A non-zero discriminant, an irrational one, and a parametric one all
        // stay undecided-or-non-zero, so the two-root branch is kept.
        assert!(!is_zero_value(pool.integer(8_i32), &pool));
        assert!(!is_zero_value(pool.symbol("a", Domain::Real), &pool));
        // √0 is zero, but only the structural arm can see it.
        assert!(is_zero_value(pool.func("sqrt", vec![zero]), &pool));
    }

    #[test]
    fn conjugate_roots_behind_a_nested_radical_both_survive() {
        // x·y = 0 and x² − y + 1 = 0.  y = 0 forces x² = −1, so (±i, 0) are
        // both solutions, alongside (0, 1).  The y value reaches the inner
        // discriminant as √1 rather than as 1, and an enclosure that lets that
        // leak a spurious imaginary width can no longer tell +i from −i.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let eq1 = pool.mul(vec![x, y]);
        let eq2 = pool.add(vec![
            powk(&pool, x, 2),
            pool.mul(vec![pool.integer(-1_i32), y]),
            pool.integer(1_i32),
        ]);
        let sols = finite(vec![eq1, eq2], vec![x, y], &pool);
        assert_eq!(sols.len(), 3, "(0,1) and (±i,0), got {sols:?}");
    }

    #[test]
    fn repeated_root_is_one_solution() {
        // The solution *set* of x² = 0 is {0} — one element, not ±√0.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let sols = finite(vec![powk(&pool, x, 2)], vec![x], &pool);
        assert_eq!(sols.len(), 1, "{sols:?}");
        assert!(eval_no_env(sols[0][0], &pool).abs() < 1e-12);
    }

    #[test]
    fn repeated_roots_do_not_multiply_across_variables() {
        // x² = y² = z² = 0 has the single solution (0,0,0); the duplicate
        // ±√0 entries used to multiply out to eight copies of the origin.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let sols = finite(
            vec![powk(&pool, x, 2), powk(&pool, y, 2), powk(&pool, z, 2)],
            vec![x, y, z],
            &pool,
        );
        assert_eq!(sols.len(), 1, "{sols:?}");
    }

    #[test]
    fn shifted_double_root_is_one_solution() {
        // (x−1)² = 0: one solution, x = 1.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let shifted = pool.add(vec![x, pool.integer(-1_i32)]);
        let sols = finite(vec![powk(&pool, shifted, 2)], vec![x], &pool);
        assert_eq!(sols.len(), 1, "{sols:?}");
        assert!((eval_no_env(sols[0][0], &pool) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn parametric_quadratic_free_rhs() {
        // x² − y = 0 in [x] → x = ±√y
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let two = pool.integer(2_i32);
        let x2 = pool.pow(x, two);
        let eq = pool.add(vec![x2, pool.mul(vec![pool.integer(-1_i32), y])]);
        let result = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();
        let SolutionSet::Finite(sols) = result else {
            panic!("expected finite parametric solutions");
        };
        assert_eq!(sols.len(), 2);
        // Bind y = 4 and check numeric roots ±2.
        let mut env = HashMap::new();
        env.insert(y, 4.0);
        let vals: Vec<f64> = sols
            .iter()
            .map(|s| eval_interp(s[0], &env, &pool).expect("eval"))
            .collect();
        assert!(vals.iter().any(|v| (v - 2.0).abs() < 1e-10));
        assert!(vals.iter().any(|v| (v + 2.0).abs() < 1e-10));
    }

    #[test]
    fn parametric_linear_affine() {
        // a·x − b = 0 in [x] → x = b/a
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let eq = pool.add(vec![
            pool.mul(vec![a, x]),
            pool.mul(vec![pool.integer(-1_i32), b]),
        ]);
        let result = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();
        let SolutionSet::Finite(sols) = result else {
            panic!("expected finite parametric solution");
        };
        assert_eq!(sols.len(), 1);
        let mut env = HashMap::new();
        env.insert(a, 2.0);
        env.insert(b, 6.0);
        let val = eval_interp(sols[0][0], &env, &pool).expect("eval");
        assert!((val - 3.0).abs() < 1e-10);
    }

    /// `b/a` is the solution **for `a ≠ 0`**. At `a = 0` the equation reads
    /// `−b = 0`, which has no solution for `b ≠ 0` and every `x` for `b = 0`;
    /// the returned tuple is a number for neither. The answer is defensible
    /// under the generic-parameter reading and indefensible unstated, and a
    /// parametric tuple is returned unverified, so the hypothesis is the only
    /// signal the caller gets.
    #[test]
    fn a_parametric_division_states_its_non_vanishing_hypothesis() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let eq = pool.add(vec![
            pool.mul(vec![a, x]),
            pool.mul(vec![pool.integer(-1_i32), b]),
        ]);
        let _ = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();

        let conds = take_solve_side_conditions();
        assert_eq!(conds.len(), 1, "{conds:?}");
        let crate::deriv::log::SideCondition::NonZero(id) = conds[0] else {
            panic!("expected a non-vanishing hypothesis, got {:?}", conds[0]);
        };
        assert_eq!(id, a);
        // Consuming: one call's hypotheses cannot be read as the next call's.
        assert!(take_solve_side_conditions().is_empty());
    }

    /// The control: a system whose leading coefficients are *proved* non-zero
    /// carries no hypothesis. Without this, "state a condition always" would
    /// pass the test above and say nothing.
    #[test]
    fn a_solve_that_proves_its_divisors_states_nothing() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        // 2x − b = 0: the divisor is the literal 2, and b is still a parameter,
        // so this is the nearest neighbour of the case above.
        let eq = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.mul(vec![pool.integer(-1_i32), b]),
        ]);
        let result = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap();
        assert!(matches!(result, SolutionSet::Finite(ref s) if s.len() == 1));
        assert!(take_solve_side_conditions().is_empty());
    }

    #[test]
    fn parametric_system_line_with_parameter() {
        // x + y − c = 0, x − y = 0 in [x, y] → x = y = c/2
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let c = pool.symbol("c", Domain::Real);
        let neg_one = pool.integer(-1_i32);
        let eq1 = pool.add(vec![x, y, pool.mul(vec![neg_one, c])]);
        let eq2 = pool.add(vec![x, pool.mul(vec![neg_one, y])]);
        let result = solve_polynomial_system(vec![eq1, eq2], vec![x, y], &pool).unwrap();
        let SolutionSet::Finite(sols) = result else {
            panic!("expected finite parametric solutions");
        };
        assert_eq!(sols.len(), 1);
        let mut env = HashMap::new();
        env.insert(c, 4.0);
        let xv = eval_interp(sols[0][0], &env, &pool).expect("eval x");
        let yv = eval_interp(sols[0][1], &env, &pool).expect("eval y");
        assert!((xv - 2.0).abs() < 1e-10);
        assert!((yv - 2.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Rational equations: clearing denominators, and taking the excess back out
    // -----------------------------------------------------------------------

    /// `a − b` as an expression.
    fn minus(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
        pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
    }

    /// `a / b` written the way a caller writes it: `a · b⁻¹`.
    fn over(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
        pool.mul(vec![a, pool.pow(b, pool.integer(-1_i32))])
    }

    /// The RC divider in the form nodal analysis produces it. Solving it at all
    /// is the point: this was `E-SOLVE-001` until rational equations were
    /// accepted, and the caller had to clear `1/R1` by hand.
    #[test]
    fn rc_divider_solves_in_admittance_form() {
        let pool = ExprPool::new();
        let vo = pool.symbol("Vo", Domain::Real);
        let vin = pool.symbol("Vin", Domain::Real);
        let r1 = pool.symbol("R1", Domain::Real);
        let c = pool.symbol("C", Domain::Real);
        let s = pool.symbol("s", Domain::Real);

        // (Vo − Vin)/R1 + Vo·s·C = 0
        let eq = pool.add(vec![
            over(minus(vo, vin, &pool), r1, &pool),
            pool.mul(vec![vo, s, c]),
        ]);
        let SolutionSet::Finite(sols) = solve_polynomial_system(vec![eq], vec![vo], &pool).unwrap()
        else {
            panic!("expected a finite solution set");
        };
        assert_eq!(sols.len(), 1);

        // Vo = Vin/(1 + s·R1·C).  At Vin = 1, s = 7, R1 = 2, C = 5 that is 1/71.
        let env: HashMap<ExprId, f64> = [(vin, 1.0), (s, 7.0), (r1, 2.0), (c, 5.0)]
            .into_iter()
            .collect();
        let got = eval_interp(sols[0][0], &env, &pool).expect("numeric value");
        assert!((got - 1.0 / 71.0).abs() < 1e-12, "got {got}");

        // The answer holds for R1 ≠ 0, and says so.
        let conds = take_solve_side_conditions();
        assert!(
            conds.contains(&crate::deriv::log::SideCondition::NonZero(r1)),
            "the cleared denominator R1 must be reported: {conds:?}"
        );
    }

    /// A resistance declared positive settles its own non-vanishing, so the
    /// hypothesis is discharged rather than reported.
    #[test]
    fn a_positive_denominator_needs_no_hypothesis() {
        let pool = ExprPool::new();
        let vo = pool.symbol("Vo", Domain::Real);
        let vin = pool.symbol("Vin", Domain::Real);
        let r1 = pool.symbol("R1", Domain::Positive);
        let c = pool.symbol("C", Domain::Real);
        let s = pool.symbol("s", Domain::Real);
        let eq = pool.add(vec![
            over(minus(vo, vin, &pool), r1, &pool),
            pool.mul(vec![vo, s, c]),
        ]);
        let SolutionSet::Finite(sols) = solve_polynomial_system(vec![eq], vec![vo], &pool).unwrap()
        else {
            panic!("expected a finite solution set");
        };
        assert_eq!(sols.len(), 1);
        let conds = take_solve_side_conditions();
        assert!(
            !conds.contains(&crate::deriv::log::SideCondition::NonZero(r1)),
            "R1 > 0 already excludes R1 = 0: {conds:?}"
        );
    }

    /// Two-node modified nodal analysis: two rational equations, two unknowns,
    /// five symbolic parameters.
    #[test]
    fn two_node_mna_system() {
        let pool = ExprPool::new();
        let v1 = pool.symbol("V1", Domain::Real);
        let v2 = pool.symbol("V2", Domain::Real);
        let vin = pool.symbol("Vin", Domain::Real);
        let r1 = pool.symbol("R1", Domain::Real);
        let r2 = pool.symbol("R2", Domain::Real);
        let c = pool.symbol("C", Domain::Real);
        let s = pool.symbol("s", Domain::Real);

        // (V1 − Vin)/R1 + (V1 − V2)/R2 = 0
        let node1 = pool.add(vec![
            over(minus(v1, vin, &pool), r1, &pool),
            over(minus(v1, v2, &pool), r2, &pool),
        ]);
        // (V2 − V1)/R2 + V2·s·C = 0
        let node2 = pool.add(vec![
            over(minus(v2, v1, &pool), r2, &pool),
            pool.mul(vec![v2, s, c]),
        ]);

        let SolutionSet::Finite(sols) =
            solve_polynomial_system(vec![node1, node2], vec![v1, v2], &pool).unwrap()
        else {
            panic!("expected a finite solution set");
        };
        assert_eq!(sols.len(), 1);

        // V2 = Vin/(1 + s·C·(R1+R2)) and V1 = V2·(1 + s·C·R2).
        // Vin = 1, R1 = 2, R2 = 3, C = 5, s = 7  →  V2 = 1/176, V1 = 53/88.
        let env: HashMap<ExprId, f64> = [(vin, 1.0), (r1, 2.0), (r2, 3.0), (c, 5.0), (s, 7.0)]
            .into_iter()
            .collect();
        let got1 = eval_interp(sols[0][0], &env, &pool).expect("V1");
        let got2 = eval_interp(sols[0][1], &env, &pool).expect("V2");
        assert!((got1 - 53.0 / 88.0).abs() < 1e-12, "V1 = {got1}");
        assert!((got2 - 1.0 / 176.0).abs() < 1e-12, "V2 = {got2}");

        let conds = take_solve_side_conditions();
        assert!(
            !conds.is_empty(),
            "R1 and R2 were divided by; that has to be visible"
        );
    }

    /// `x/(x−1) = 1/(x−1)` clears to `(x−1)² = 0`. Its root `x = 1` is exactly
    /// where the original equation reads `1/0 = 1/0`, so it is **not** a
    /// solution — returning it would be a wrong answer, not a rough edge.
    #[test]
    fn a_root_that_kills_the_denominator_is_not_returned() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let xm1 = minus(x, pool.integer(1_i32), &pool);
        let eq = minus(
            over(x, xm1, &pool),
            over(pool.integer(1_i32), xm1, &pool),
            &pool,
        );
        match solve_polynomial_system(vec![eq], vec![x], &pool).unwrap() {
            SolutionSet::Finite(sols) => assert!(
                sols.is_empty(),
                "x = 1 is a pole of both sides, not a solution: {sols:?}"
            ),
            SolutionSet::NoSolution => {}
            SolutionSet::Parametric(_) => panic!("expected a decided answer"),
        }
    }

    /// `1/x = 0` has no solution. Not `x = ∞`, and not the `x = 0` a careless
    /// cancellation would produce.
    #[test]
    fn a_reciprocal_is_never_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.pow(x, pool.integer(-1_i32));
        match solve_polynomial_system(vec![eq], vec![x], &pool).unwrap() {
            SolutionSet::NoSolution => {}
            SolutionSet::Finite(sols) => {
                assert!(sols.is_empty(), "1/x = 0 has no solution: {sols:?}")
            }
            SolutionSet::Parametric(_) => panic!("expected a decided answer"),
        }
    }

    /// `x²/x = 0` is the case reducing to lowest terms gets wrong: cancelling
    /// gives `x = 0`, but the original expression is `0/0` there.
    #[test]
    fn a_removable_singularity_is_still_not_a_solution() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = over(pool.pow(x, pool.integer(2_i32)), x, &pool);
        match solve_polynomial_system(vec![eq], vec![x], &pool).unwrap() {
            SolutionSet::Finite(sols) => {
                assert!(sols.is_empty(), "x = 0 makes it read 0/0: {sols:?}")
            }
            SolutionSet::NoSolution => {}
            SolutionSet::Parametric(_) => panic!("expected a decided answer"),
        }
    }

    /// `1/(1/x − 1) = 0` has no solution, and the reason is the one a "product
    /// of the denominators" reading loses.
    ///
    /// The reciprocal swaps the halves — `(n/d)⁻¹ = d/n` — so the inner
    /// denominator `x` becomes the outer *numerator* and the requirement
    /// `x ≠ 0` leaves the denominator product entirely. What is left, `1 − x`,
    /// is perfectly non-zero at `x = 0`, so the cleared numerator's only root
    /// sails through and `x = 0` comes back as a solution of an equation that
    /// has no value there.
    #[test]
    fn a_condition_hidden_by_a_reciprocal_is_not_lost() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let inner = minus(
            pool.pow(x, pool.integer(-1_i32)),
            pool.integer(1_i32),
            &pool,
        );
        let eq = pool.pow(inner, pool.integer(-1_i32));
        match solve_polynomial_system(vec![eq], vec![x], &pool).unwrap() {
            SolutionSet::Finite(sols) => assert!(
                sols.is_empty(),
                "1/x is undefined at x = 0, so the whole equation is: {sols:?}"
            ),
            SolutionSet::NoSolution => {}
            SolutionSet::Parametric(_) => panic!("expected a decided answer"),
        }
    }

    /// One equation's root is another's pole: the system has no solution even
    /// though the cleared numerator system has a perfectly good one.
    #[test]
    fn a_pole_of_a_sibling_equation_excludes_the_root() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let one = pool.integer(1_i32);
        let ym1 = minus(y, one, &pool);
        // (x − 1)/(y − 1) = 0,  y − 1 = 0
        let eq1 = over(minus(x, one, &pool), ym1, &pool);
        match solve_polynomial_system(vec![eq1, ym1], vec![x, y], &pool).unwrap() {
            SolutionSet::Finite(sols) => assert!(
                sols.is_empty(),
                "(1,1) is a pole of the first equation: {sols:?}"
            ),
            SolutionSet::NoSolution => {}
            SolutionSet::Parametric(_) => panic!("expected a decided answer"),
        }
    }

    /// A rational equation whose roots are genuine keeps all of them; the pole
    /// it does have is simply not one of them.
    #[test]
    fn genuine_roots_of_a_rational_equation_survive() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        // (x² − 4)/(x − 1) = 0  →  x = ±2
        let num = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-4_i32)]);
        let eq = over(num, minus(x, one, &pool), &pool);
        let SolutionSet::Finite(sols) = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap()
        else {
            panic!("expected a finite solution set");
        };
        assert_eq!(sols.len(), 2);
        let mut vals: Vec<f64> = sols.iter().map(|s| eval_no_env(s[0], &pool)).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] + 2.0).abs() < 1e-10 && (vals[1] - 2.0).abs() < 1e-10);
        // No parameter is involved, so nothing had to be assumed.
        assert!(take_solve_side_conditions().is_empty());
    }

    /// Widening to rational functions must not widen to transcendental ones:
    /// `exp(x) − 2` keeps refusing, with the code it always had.
    #[test]
    fn a_transcendental_still_refuses_with_its_own_code() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.add(vec![pool.func("exp", vec![x]), pool.integer(-2_i32)]);
        let Err(err) = solve_polynomial_system(vec![eq], vec![x], &pool) else {
            panic!("a transcendental must not be solved by the polynomial path");
        };
        assert!(matches!(err, SolverError::NotPolynomial(_)), "{err}");
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-SOLVE-001");
    }

    /// An equation whose denominator is identically zero denotes no function;
    /// clearing it would multiply by zero and make every point a "solution".
    #[test]
    fn an_everywhere_undefined_equation_is_refused() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = minus(x, x, &pool);
        let eq = pool.pow(zero, pool.integer(-1_i32));
        let Err(err) = solve_polynomial_system(vec![eq], vec![x], &pool) else {
            panic!("an everywhere-undefined equation must not be solved");
        };
        assert!(matches!(err, SolverError::NotPolynomial(_)), "{err}");
        let refusal = take_undefined_equation().expect("refusal recorded out of band");
        assert_eq!(crate::errors::AlkahestError::code(&refusal), "E-SOLVE-005");
    }

    /// A genuinely transcendental refusal must never be re-attributed to the
    /// undefined-equation carrier left behind by an earlier call.
    #[test]
    fn an_undefined_equation_refusal_does_not_leak_into_the_next_solve() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let zero = minus(x, x, &pool);
        let bad = pool.pow(zero, pool.integer(-1_i32));
        assert!(solve_polynomial_system(vec![bad], vec![x], &pool).is_err());

        let trans = pool.add(vec![pool.func("exp", vec![x]), pool.integer(-2_i32)]);
        assert!(solve_polynomial_system(vec![trans], vec![x], &pool).is_err());
        assert!(
            take_undefined_equation().is_none(),
            "exp(x) − 2 is not an undefined equation"
        );
    }

    /// Polynomial input takes exactly the path it always did: no denominator,
    /// no exclusion step, no hypothesis invented.
    #[test]
    fn polynomial_input_records_no_denominator_hypothesis() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let eq = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(-4_i32)]);
        let SolutionSet::Finite(sols) = solve_polynomial_system(vec![eq], vec![x], &pool).unwrap()
        else {
            panic!("expected a finite solution set");
        };
        assert_eq!(sols.len(), 2);
        assert!(take_solve_side_conditions().is_empty());
    }

    // -----------------------------------------------------------------------
    // G1 — degree ≥ 3: factor first, then formula / radicals / RootOf
    // -----------------------------------------------------------------------

    /// `Σ cₖ·xᵏ` from small integer coefficients (ascending).
    fn upoly(x: ExprId, c: &[i64], pool: &ExprPool) -> ExprId {
        let terms: Vec<ExprId> = c
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(k, &v)| match k {
                0 => pool.integer(v),
                1 => pool.mul(vec![pool.integer(v), x]),
                _ => pool.mul(vec![pool.integer(v), pool.pow(x, pool.integer(k as i64))]),
            })
            .collect();
        pool.add(terms)
    }

    /// Every value as a complex number, via the evaluator `numeric=True` uses.
    fn complex_values(sols: &[Solution], pool: &ExprPool) -> Vec<(f64, f64)> {
        sols.iter()
            .map(|s| {
                let c = crate::eval::eval_complex_f64(s[0], pool, &HashMap::new())
                    .unwrap_or_else(|e| panic!("{}: {e:?}", pool.display(s[0])));
                (c.re, c.im)
            })
            .collect()
    }

    /// `|p(z)|` small relative to `Σ|cₖ||z|ᵏ` for every value.
    fn assert_roots_of(c: &[i64], vals: &[(f64, f64)]) {
        for &(re, im) in vals {
            let (mut ar, mut ai, mut mag) = (0.0f64, 0.0f64, 0.0f64);
            let az = re.hypot(im);
            for &k in c.iter().rev() {
                let nr = ar * re - ai * im + k as f64;
                ai = ar * im + ai * re;
                ar = nr;
                mag = mag * az + (k as f64).abs();
            }
            assert!(
                ar.hypot(ai) <= 1e-9 * mag.max(1.0),
                "({re}, {im}) is not a root of {c:?}"
            );
        }
    }

    #[test]
    fn a_cubic_with_three_rational_roots_is_solved() {
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Real);
        let sols = finite(vec![upoly(s, &[6, 11, 6, 1], &pool)], vec![s], &pool);
        let mut got: Vec<ExprId> = sols.iter().map(|v| v[0]).collect();
        got.sort();
        let mut want = vec![pool.integer(-1), pool.integer(-2), pool.integer(-3)];
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn report_repros_return_every_root() {
        for c in [
            vec![0, -5, 0, 1],               // s(s² − 5)
            vec![0, -3, 2, 1],               // x³ + 2x² − 3x
            vec![1, 0, 0, 0, 1],             // s⁴ + 1 (Butterworth, n = 4)
            vec![1, 0, 0, 0, 0, 0, 0, 0, 1], // s⁸ + 1
            vec![0, -3, 0, 5],               // 2·P₃
            vec![3, 0, -30, 0, 35],          // 8·P₄
        ] {
            let pool = ExprPool::new();
            let x = pool.symbol("x", Domain::Real);
            let sols = finite(vec![upoly(x, &c, &pool)], vec![x], &pool);
            assert_eq!(sols.len(), c.len() - 1, "{c:?}");
            let vals = complex_values(&sols, &pool);
            assert_roots_of(&c, &vals);
            // Distinct.
            for i in 0..vals.len() {
                for j in 0..i {
                    let d = (vals[i].0 - vals[j].0).hypot(vals[i].1 - vals[j].1);
                    assert!(d > 1e-6, "{c:?}: repeated value {:?}", vals[i]);
                }
            }
        }
    }

    #[test]
    fn a_repeated_root_is_reported_once_at_degree_three() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x − 1)²(x + 2) = x³ − 3x + 2
        let sols = finite(vec![upoly(x, &[2, -3, 0, 1], &pool)], vec![x], &pool);
        let mut got: Vec<ExprId> = sols.iter().map(|v| v[0]).collect();
        got.sort();
        let mut want = vec![pool.integer(1), pool.integer(-2)];
        want.sort();
        assert_eq!(got, want);
    }

    #[cfg(flint_arb)]
    #[test]
    fn an_irreducible_quintic_gives_root_of() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let c = [-1, -1, 0, 0, 0, 1];
        let sols = finite(vec![upoly(x, &c, &pool)], vec![x], &pool);
        assert_eq!(sols.len(), 5);
        for (k, s) in sols.iter().enumerate() {
            assert_eq!(
                pool.display(s[0]).to_string(),
                format!("RootOf(x^5 - x - 1, {k})")
            );
        }
        let vals = complex_values(&sols, &pool);
        assert_roots_of(&c, &vals);
        // Index 0 is the one real root.
        assert!(vals[0].1 == 0.0 && (vals[0].0 - 1.1673039782614187).abs() < 1e-14);
    }

    #[test]
    fn a_parametric_cubic_with_a_rational_factor_is_solved() {
        // s³ − a·s² − s + a = (s − a)(s − 1)(s + 1)
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let s2 = pool.pow(s, pool.integer(2));
        let eq = pool.add(vec![
            pool.pow(s, pool.integer(3)),
            pool.mul(vec![pool.integer(-1), a, s2]),
            pool.mul(vec![pool.integer(-1), s]),
            a,
        ]);
        let sols = finite(vec![eq], vec![s], &pool);
        let got: std::collections::BTreeSet<String> = sols
            .iter()
            .map(|v| {
                let e = crate::simplify::engine::simplify(v[0], &pool).value;
                pool.display(e).to_string()
            })
            .collect();
        assert_eq!(sols.len(), 3, "{got:?}");
        assert!(
            got.contains("a") && got.contains("1") && got.contains("-1"),
            "{got:?}"
        );
    }

    #[test]
    fn a_parametric_irreducible_cubic_still_refuses_with_e_solve_002() {
        // s³ + a·s + 1: irreducible over ℚ(a), no closed form here.
        let pool = ExprPool::new();
        let s = pool.symbol("s", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let eq = pool.add(vec![
            pool.pow(s, pool.integer(3)),
            pool.mul(vec![a, s]),
            pool.integer(1),
        ]);
        let err = solve_polynomial_system(vec![eq], vec![s], &pool)
            .err()
            .expect("refused");
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-SOLVE-002");
    }

    #[test]
    fn a_multivariate_system_back_substitutes_through_a_cubic() {
        // e₁ = 6, e₂ = 11, e₃ = 6: the permutations of (1, 2, 3).
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let e1 = pool.add(vec![x, y, z, pool.integer(-6)]);
        let e2 = pool.add(vec![
            pool.mul(vec![x, y]),
            pool.mul(vec![y, z]),
            pool.mul(vec![z, x]),
            pool.integer(-11),
        ]);
        let e3 = pool.add(vec![pool.mul(vec![x, y, z]), pool.integer(-6)]);
        let sols = finite(vec![e1, e2, e3], vec![x, y, z], &pool);
        let mut got: Vec<[i64; 3]> = sols
            .iter()
            .map(|s| {
                let v: Vec<i64> = s
                    .iter()
                    .map(|&e| eval_no_env(e, &pool).round() as i64)
                    .collect();
                [v[0], v[1], v[2]]
            })
            .collect();
        got.sort();
        assert_eq!(
            got,
            vec![
                [1, 2, 3],
                [1, 3, 2],
                [2, 1, 3],
                [2, 3, 1],
                [3, 1, 2],
                [3, 2, 1]
            ]
        );
    }

    #[cfg(flint_arb)]
    #[test]
    fn a_root_of_coordinate_back_substitutes() {
        // ⟨x − y², y³ − y − 1⟩: y is a RootOf, x = y².
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e1 = pool.add(vec![
            x,
            pool.mul(vec![pool.integer(-1), pool.pow(y, pool.integer(2))]),
        ]);
        let e2 = upoly(y, &[-1, -1, 0, 1], &pool);
        let sols = finite(vec![e1, e2], vec![x, y], &pool);
        assert_eq!(sols.len(), 3);
        let env = HashMap::new();
        for s in &sols {
            let xv = crate::eval::eval_complex_f64(s[0], &pool, &env).expect("x evaluates");
            let yv = crate::eval::eval_complex_f64(s[1], &pool, &env).expect("y evaluates");
            let (y2r, y2i) = (yv.re * yv.re - yv.im * yv.im, 2.0 * yv.re * yv.im);
            assert!((xv.re - y2r).hypot(xv.im - y2i) < 1e-12);
            let (y3r, y3i) = (y2r * yv.re - y2i * yv.im, y2r * yv.im + y2i * yv.re);
            assert!((y3r - yv.re - 1.0).hypot(y3i - yv.im) < 1e-12);
        }
    }

    /// B4 (benchmark 2026-10-25): the 3.11.0 wheel refused
    /// `⟨x² + y² + z² − 4, xy − z − 1, x + y + z³ − 2⟩` with `E-SOLVE-002`
    /// because its eliminant in `z` has degree 6. All 12 complex solutions
    /// (Bézout 2·2·3, confirmed by `NSolve`) come back exactly, `z` as a
    /// `RootOf` of the irreducible sextic, and each one satisfies the system.
    #[cfg(flint_arb)]
    #[test]
    fn a_sextic_eliminant_gives_every_solution_exactly() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let n = |v: i64| pool.integer(v);
        let sq = |e: ExprId| pool.pow(e, pool.integer(2));
        let e1 = pool.add(vec![sq(x), sq(y), sq(z), n(-4)]);
        let e2 = pool.add(vec![pool.mul(vec![x, y]), pool.mul(vec![n(-1), z]), n(-1)]);
        let e3 = pool.add(vec![x, y, pool.pow(z, pool.integer(3)), n(-2)]);
        let sols = finite(vec![e1, e2, e3], vec![x, y, z], &pool);
        assert_eq!(sols.len(), 12);
        let env = HashMap::new();
        let c = |e: ExprId| {
            let v = crate::eval::eval_complex_f64(e, &pool, &env)
                .unwrap_or_else(|err| panic!("{}: {err:?}", pool.display(e)));
            (v.re, v.im)
        };
        let mul = |a: (f64, f64), b: (f64, f64)| (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0);
        let mut points: Vec<[(f64, f64); 3]> = Vec::new();
        for s in &sols {
            assert!(
                pool.display(s[2]).to_string().contains("RootOf"),
                "z = {}",
                pool.display(s[2])
            );
            let (xv, yv, zv) = (c(s[0]), c(s[1]), c(s[2]));
            let (x2, y2, z2) = (mul(xv, xv), mul(yv, yv), mul(zv, zv));
            let xy = mul(xv, yv);
            let z3 = mul(z2, zv);
            let r1 = (x2.0 + y2.0 + z2.0 - 4.0, x2.1 + y2.1 + z2.1);
            let r2 = (xy.0 - zv.0 - 1.0, xy.1 - zv.1);
            let r3 = (xv.0 + yv.0 + z3.0 - 2.0, xv.1 + yv.1 + z3.1);
            for r in [r1, r2, r3] {
                assert!(r.0.hypot(r.1) < 1e-9, "residual {r:?}");
            }
            assert!(
                points.iter().all(|p| {
                    p.iter()
                        .zip([xv, yv, zv])
                        .map(|(a, b)| (a.0 - b.0).hypot(a.1 - b.1))
                        .sum::<f64>()
                        > 1e-6
                }),
                "duplicate solution"
            );
            points.push([xv, yv, zv]);
        }
        // The two real solutions the benchmark's numeric solve reported.
        let real = points
            .iter()
            .filter(|p| p.iter().all(|v| v.1.abs() < 1e-12))
            .count();
        assert_eq!(real, 2);
        assert!(points.iter().any(|p| {
            (p[0].0 - 0.242_67).abs() < 1e-5
                && (p[1].0 - 1.911_45).abs() < 1e-5
                && (p[2].0 + 0.536_15).abs() < 1e-5
        }));
    }

    // -----------------------------------------------------------------------
    // Parametric systems: branches decided as functions of the parameters
    // -----------------------------------------------------------------------

    fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
        pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
    }

    /// Every tuple, at the parameter values in `env`, as `(x, y)` floats.
    fn points(sols: &[Solution], env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Vec<(f64, f64)> {
        sols.iter()
            .map(|s| {
                (
                    eval_interp(s[0], env, pool).expect("x evaluates"),
                    eval_interp(s[1], env, pool).expect("y evaluates"),
                )
            })
            .collect()
    }

    fn same_set(got: &[(f64, f64)], want: &[(f64, f64)]) -> bool {
        let close = |p: &(f64, f64), q: &(f64, f64)| (p.0 - q.0).abs() + (p.1 - q.1).abs() < 1e-9;
        got.len() == want.len()
            && got.iter().all(|p| want.iter().any(|q| close(p, q)))
            && want.iter().all(|q| got.iter().any(|p| close(p, q)))
    }

    fn lotka_volterra(pool: &ExprPool, domain: Domain) -> (Vec<ExprId>, [ExprId; 6]) {
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let [a, b, c, d] = ["a", "b", "c", "d"].map(|n| pool.symbol(n, domain));
        // a·x − b·x·y,  −c·y + d·x·y
        let e1 = sub(pool.mul(vec![a, x]), pool.mul(vec![b, x, y]), pool);
        let e2 = sub(pool.mul(vec![d, x, y]), pool.mul(vec![c, y]), pool);
        (vec![e1, e2], [x, y, a, b, c, d])
    }

    /// N2: the Lotka–Volterra equilibria are `(0, 0)` and `(c/d, a/b)`.  The
    /// back-substitution used to divide by `b·y − a` at `y = a/b` and return
    /// `(0, (ac + √(a²c²))/(2bc))` — not a solution — and lose `(c/d, a/b)`.
    #[test]
    fn lotka_volterra_equilibria_are_both_found_and_both_solutions() {
        let pool = ExprPool::new();
        let (eqs, [x, y, a, b, c, d]) = lotka_volterra(&pool, Domain::Real);
        let sols = finite(eqs, vec![x, y], &pool);
        let env: HashMap<ExprId, f64> = [(a, 2.0), (b, 1.0), (c, 3.0), (d, 1.5)].into();
        let got = points(&sols, &env, &pool);
        assert!(same_set(&got, &[(0.0, 0.0), (2.0, 2.0)]), "{got:?}");
        // The hypotheses are the divisors of the kept solution, `b` and `d`,
        // and nothing from a discarded branch (it used to list `(d·0) ≠ 0`).
        let conds = take_solve_side_conditions();
        use crate::deriv::log::SideCondition::NonZero;
        assert_eq!(conds.len(), 2, "{conds:?}");
        assert!(
            conds.contains(&NonZero(b)) && conds.contains(&NonZero(d)),
            "{conds:?}"
        );
    }

    /// With positive parameters the same system used to return `x = −1·0·0⁻¹`.
    #[test]
    fn lotka_volterra_with_positive_parameters_has_no_zero_division() {
        let pool = ExprPool::new();
        let (eqs, [x, y, a, b, c, d]) = lotka_volterra(&pool, Domain::Positive);
        let sols = finite(eqs, vec![x, y], &pool);
        let env: HashMap<ExprId, f64> = [(a, 2.0), (b, 1.0), (c, 3.0), (d, 1.5)].into();
        let got = points(&sols, &env, &pool);
        assert!(same_set(&got, &[(0.0, 0.0), (2.0, 2.0)]), "{got:?}");
        // `b > 0` and `d > 0` discharge both divisions.
        assert!(take_solve_side_conditions().is_empty());
    }

    /// `(x − a)(y − b) = (x − c)(y − d) = 0` has exactly `(a, d)` and `(c, b)`
    /// for generic parameters; it used to return a `0/0` point with residual 1.
    #[test]
    fn two_hyperbolas_meet_in_two_points() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let [a, b, c, d] = ["a", "b", "c", "d"].map(|n| pool.symbol(n, Domain::Real));
        let e1 = pool.mul(vec![sub(x, a, &pool), sub(y, b, &pool)]);
        let e2 = pool.mul(vec![sub(x, c, &pool), sub(y, d, &pool)]);
        let sols = finite(vec![e1, e2], vec![x, y], &pool);
        let env: HashMap<ExprId, f64> = [(a, 1.0), (b, 2.0), (c, 3.0), (d, 5.0)].into();
        let got = points(&sols, &env, &pool);
        assert!(same_set(&got, &[(1.0, 5.0), (3.0, 2.0)]), "{got:?}");
    }

    /// `x(y − a) = x(y − b) = 0` is the line `x = 0` for `a ≠ b`.  It used to
    /// collapse to a single `0/0` point; a positive-dimensional component is
    /// returned as the ideal instead.
    #[test]
    fn a_parametric_line_is_not_collapsed_to_a_point() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let e1 = pool.mul(vec![x, sub(y, a, &pool)]);
        let e2 = pool.mul(vec![x, sub(y, b, &pool)]);
        let out = solve_polynomial_system(vec![e1, e2], vec![x, y], &pool).unwrap();
        assert!(matches!(out, SolutionSet::Parametric(_)));
    }

    /// `a·x·(1 − y) = 0, y = 2` is `x = 0` — for `a ≠ 0`; at `a = 0` every `x`
    /// solves it.  Dropping the factor `a` from the branches is an assumption,
    /// and it is stated.
    #[test]
    fn a_dropped_parameter_factor_is_a_stated_hypothesis() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let one_minus_y = sub(pool.integer(1_i32), y, &pool);
        let e1 = pool.mul(vec![a, x, one_minus_y]);
        let e2 = sub(y, pool.integer(2_i32), &pool);
        let sols = finite(vec![e1, e2], vec![x, y], &pool);
        assert_eq!(sols.len(), 1);
        assert!(eval_no_env(sols[0][0], &pool).abs() < 1e-12);
        assert!((eval_no_env(sols[0][1], &pool) - 2.0).abs() < 1e-12);
        let conds = take_solve_side_conditions();
        assert_eq!(conds, vec![crate::deriv::log::SideCondition::NonZero(a)]);
    }

    /// `x = a ∧ x = b` has no solution unless `a = b`; it used to return one.
    #[test]
    fn a_condition_on_the_parameters_alone_is_generic_inconsistency() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let out = solve_polynomial_system(vec![sub(x, a, &pool), sub(x, b, &pool)], vec![x], &pool)
            .unwrap();
        assert!(matches!(out, SolutionSet::NoSolution));
        assert_eq!(take_solve_side_conditions().len(), 1);
    }

    /// The SEIR disease-free equilibrium is `S = N, E = I = 0`.  It used to be
    /// reported with the endemic `S*` because the coefficient of `S` that
    /// vanishes at `I = 0` was assumed non-zero.
    #[test]
    fn seir_disease_free_equilibrium_is_s_equals_n() {
        let pool = ExprPool::new();
        let [beta, gamma, sigma, mu, n] =
            ["beta", "gamma", "sigma", "mu", "N"].map(|s| pool.symbol(s, Domain::Positive));
        let [s, e, i] = ["S", "E", "I"].map(|v| pool.symbol(v, Domain::Real));
        let neg = |t: ExprId| pool.mul(vec![pool.integer(-1_i32), t]);
        let inv_n = pool.pow(n, pool.integer(-1_i32));
        let infect = pool.mul(vec![beta, s, i, inv_n]);
        let f_s = pool.add(vec![
            pool.mul(vec![mu, n]),
            neg(infect),
            neg(pool.mul(vec![mu, s])),
        ]);
        let f_e = pool.add(vec![
            infect,
            neg(pool.mul(vec![pool.add(vec![sigma, mu]), e])),
        ]);
        let f_i = pool.add(vec![
            pool.mul(vec![sigma, e]),
            neg(pool.mul(vec![pool.add(vec![gamma, mu]), i])),
        ]);
        let sols = finite(vec![f_s, f_e, f_i], vec![s, e, i], &pool);
        let env: HashMap<ExprId, f64> = [
            (beta, 0.5),
            (sigma, 0.2),
            (gamma, 0.1),
            (mu, 0.01),
            (n, 1000.0),
        ]
        .into();
        let got: Vec<[f64; 3]> = sols
            .iter()
            .map(|t| [0, 1, 2].map(|k| eval_interp(t[k], &env, &pool).expect("value")))
            .collect();
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(
            got.iter()
                .any(|v| (v[0] - 1000.0).abs() < 1e-9 && v[1].abs() < 1e-12 && v[2].abs() < 1e-12),
            "{got:?}"
        );
        assert!(got.iter().any(|v| (v[0] - 231.0).abs() < 1e-9), "{got:?}");
    }

    // -----------------------------------------------------------------------
    // N4: the unknown's declared domain
    // -----------------------------------------------------------------------

    #[test]
    fn a_positive_unknown_excludes_negative_roots() {
        let pool = ExprPool::new();
        let xp = pool.symbol("xp", Domain::Positive);
        let three = pool.integer(3_i32);
        assert!(finite(vec![pool.add(vec![xp, three])], vec![xp], &pool).is_empty());
        let four = pool.integer(4_i32);
        let sols = finite(vec![sub(powk(&pool, xp, 2), four, &pool)], vec![xp], &pool);
        assert_eq!(sols.len(), 1);
        assert!((eval_no_env(sols[0][0], &pool) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn an_integer_unknown_excludes_fractions() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", Domain::Integer);
        let two_n = pool.mul(vec![pool.integer(2_i32), n]);
        let three = pool.integer(3_i32);
        assert!(finite(vec![sub(two_n, three, &pool)], vec![n], &pool).is_empty());
        // ±2 from (0 ± √16)/2 are integers, proved through the exact value.
        let four = pool.integer(4_i32);
        let sols = finite(vec![sub(powk(&pool, n, 2), four, &pool)], vec![n], &pool);
        assert_eq!(sols.len(), 2);
        assert!(take_solve_side_conditions().is_empty());
    }

    /// Undecided membership is stated, not assumed and not silently dropped.
    #[test]
    fn an_undecided_domain_membership_is_a_side_condition() {
        let pool = ExprPool::new();
        let xp = pool.symbol("xp", Domain::Positive);
        let a = pool.symbol("a", Domain::Real);
        let sols = finite(vec![sub(xp, a, &pool)], vec![xp], &pool);
        assert_eq!(sols.len(), 1);
        let conds = take_solve_side_conditions();
        assert!(
            conds
                .iter()
                .any(|c| matches!(c, crate::deriv::log::SideCondition::Positive(_))),
            "{conds:?}"
        );
    }
}
