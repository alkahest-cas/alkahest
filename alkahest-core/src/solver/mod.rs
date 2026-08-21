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
pub mod homotopy;
pub mod polyhedral;
pub mod regular_chains;
pub mod transcendental;
mod verify;

pub use transcendental::{solve_transcendental, TranscendentalOutcome};

pub use regular_chains::{
    extract_regular_chain_from_basis, main_variable_recursive, triangularize, RegularChain,
};

pub use homotopy::{solve_numerical, CertifiedPoint, HomotopyError, HomotopyOpts};

pub use diophantine::{diophantine, DiophantineError, DiophantineSolution};

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
    /// An equation is not a polynomial in the given variables.
    NotPolynomial(String),
    /// Back-substitution would require solving a degree > 2 univariate — not yet
    /// implemented for general algebraic numbers.
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
                 (only degree ≤ 2 is currently supported)"
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
                "ensure all equations are polynomial in the declared variables; \
                 transcendental functions are not supported",
            ),
            SolverError::HighDegree(_) => Some(
                "degree > 2 univariate solving is not yet implemented symbolically; \
                 retry with numeric=True or method=\"homotopy\"",
            ),
            SolverError::ShapeMismatch => {
                Some("provide one equation per variable for zero-dimensional system solving")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expr → GbPoly conversion
// ---------------------------------------------------------------------------

/// Convert an `Expr` (which must be a polynomial in `vars`) to a `GbPoly`
/// with rational coefficients.  The variable order in the exponent vector
/// follows the order of `vars`.
pub fn expr_to_gbpoly(
    expr: ExprId,
    vars: &[ExprId],
    pool: &ExprPool,
) -> Result<GbPoly, SolverError> {
    let n = vars.len();
    expr_to_gbpoly_rec(expr, vars, n, pool)
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
                result = result.mul(&p);
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
                    let n_val = n.to_i64().unwrap_or(-1);
                    if n_val < 0 {
                        return Err(SolverError::NotPolynomial(format!(
                            "negative exponent {n_val} in polynomial"
                        )));
                    }
                    let base_poly = expr_to_gbpoly_rec(base, vars, n_vars, pool)?;
                    let mut result = GbPoly::constant(Rational::from(1), n_vars);
                    let mut cur = base_poly;
                    let mut rem = n_val as u64;
                    while rem > 0 {
                        if rem & 1 == 1 {
                            result = result.mul(&cur);
                        }
                        let cur2 = cur.clone();
                        cur = cur.mul(&cur2);
                        rem >>= 1;
                    }
                    Ok(result)
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
    expr_to_param_gbpoly_rec(expr, vars, params, pool)
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
                let cur2 = cur.clone();
                cur = cur.mul(&cur2);
                rem >>= 1;
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
/// is never stated — and a parametric tuple is returned *unverified* by design
/// (it is not a number, so the post-condition filter has nothing to substitute), so
/// this is the only honest signal available on that path.
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
pub fn take_solve_side_conditions() -> Vec<crate::deriv::log::SideCondition> {
    ASSUMED_NONZERO.with(|c| {
        std::mem::take(&mut *c.borrow_mut())
            .into_iter()
            .map(crate::deriv::log::SideCondition::NonZero)
            .collect()
    })
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
fn leading_is_reliable(lead: ExprId, pool: &ExprPool) -> LeadStatus {
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
        // `Unsupported` is the parametric case; `Undefined` is not a usable
        // coefficient under any reading.
        Err(verify::VerifyGap::Unsupported) => LeadStatus::AssumedNonzero,
        Err(verify::VerifyGap::Undefined) => LeadStatus::Unusable,
    }
}

/// What [`leading_is_reliable`] could establish about a leading coefficient.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LeadStatus {
    /// Proved non-zero — the step is unconditional.
    Nonzero,
    /// Not decidable here (it mentions a free parameter): usable only under the
    /// hypothesis that it does not vanish, which the caller must be told about.
    AssumedNonzero,
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
) -> Result<Option<(usize, Vec<ExprId>)>, SolverError> {
    let mut best: Option<(usize, Vec<ExprId>, u32, Option<ExprId>)> = None;
    let mut blocked_by_degree: Option<u32> = None;

    for g in gens {
        let unassigned: Vec<usize> = active_solve_vars(g, n_solve)
            .into_iter()
            .filter(|&i| partial[i].is_none())
            .collect();
        let [var_idx] = unassigned[..] else {
            continue;
        };
        let deg = max_degree_in_var(g, var_idx);
        if deg == 0 {
            continue;
        }
        if deg > 2 {
            blocked_by_degree = Some(blocked_by_degree.map_or(deg, |d: u32| d.min(deg)));
            continue;
        }
        if best.as_ref().is_some_and(|(_, _, bd, _)| *bd <= deg) {
            continue;
        }
        let coeffs: Vec<ExprId> = (0..=deg)
            .map(|k| extract_coeff_in_var(g, var_idx, k, vars, partial, pool))
            .collect();
        let lead = coeffs[deg as usize];
        let assumed = match leading_is_reliable(lead, pool) {
            LeadStatus::Unusable => continue,
            LeadStatus::Nonzero => None,
            LeadStatus::AssumedNonzero => Some(lead),
        };
        best = Some((var_idx, coeffs, deg, assumed));
    }

    match best {
        // Only the step actually taken contributes a hypothesis: generators
        // that were examined and passed over divide nothing.
        Some((var_idx, coeffs, _, assumed)) => {
            if let Some(lead) = assumed {
                assume_nonzero(lead);
            }
            Ok(Some((var_idx, coeffs)))
        }
        None => match blocked_by_degree {
            Some(d) => Err(SolverError::HighDegree(d as usize)),
            None => Ok(None),
        },
    }
}

/// Backsolve over a fixed generator list (full Gröbner basis or a triangular
/// subset).
enum BacksolveOutcome {
    Finite(Vec<Solution>),
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
    let mut partials: Vec<Vec<Option<ExprId>>> = vec![initial];

    for _ in 0..n_solve {
        let mut new_partials = Vec::new();
        let mut high_degree: Option<SolverError> = None;
        for partial in &partials {
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
            let Some((var_idx, coeffs)) = step else {
                if partial_is_refuted(gens, partial, n_solve, n_vars, pool) {
                    // A dead branch, not an under-determined one: drop it.
                    continue;
                }
                return Ok(BacksolveOutcome::Stuck);
            };
            for root in solve_univariate_symbolic(&coeffs, pool)? {
                let mut np = partial.clone();
                np[var_idx] = Some(root);
                new_partials.push(np);
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

    let solutions: Vec<Solution> = partials
        .into_iter()
        .map(|p| {
            p.into_iter()
                .take(n_solve)
                .map(|o| o.expect("all solve vars assigned"))
                .collect()
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
    n_solve: usize,
    n_vars: usize,
    pool: &ExprPool,
) -> bool {
    if n_solve != n_vars {
        return false; // parameters: no numeric residual to test
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
/// Tuples containing a free parameter are not numbers and are returned
/// unexamined; parametric solving keeps its generic-value semantics.
fn refine_solutions(
    solutions: Vec<Solution>,
    orig_polys: &[GbPoly],
    n_vars: usize,
    pool: &ExprPool,
) -> Vec<Solution> {
    let mut kept: Vec<Solution> = Vec::new();
    let mut kept_values: Vec<Vec<verify::CBall>> = Vec::new();
    let mut evaluator = verify::CBallEval::default();

    for sol in solutions {
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
            // A parameter (or any node the checker does not model): nothing can
            // be proved, so nothing is claimed — keep it as it was produced.
            Some(verify::VerifyGap::Unsupported) => {
                kept.push(sol);
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
            kept.push(sol);
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
        kept.push(sol);
    }
    kept
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

/// Solve a polynomial system in the declared unknowns.
///
/// `equations` — list of `ExprId` each representing `p = 0`.
/// `vars` — unknowns to solve for (order used for `GbPoly` exponent vectors).
///
/// Symbols that appear in `equations` but are absent from `vars` are treated as
/// free parameters: solutions may be expressions in those symbols (e.g.
/// `x² − y = 0` in `[x]` yields `x = ±√y`).
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
/// A *parametric* tuple is not a number and so cannot be checked at all: it is
/// returned unverified, under whatever non-vanishing assumptions the
/// back-substitution made about leading coefficients that mention free
/// parameters.  Those assumptions are not left unsaid — see
/// [`take_solve_side_conditions`], which must be read before the next call on
/// this thread.
pub fn solve_polynomial_system(
    equations: Vec<ExprId>,
    vars: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<SolutionSet, SolverError> {
    // Hypotheses describe *this* call; a caller reading them after it must
    // never see one left behind by an earlier solve.
    let _ = take_solve_side_conditions();
    let n_solve = vars.len();
    let params = collect_parameters(&equations, &vars, pool);
    let mut all_vars = vars;
    all_vars.extend(params);
    let n_vars = all_vars.len();

    let mut polys: Vec<GbPoly> = Vec::with_capacity(equations.len());
    for eq in &equations {
        polys.push(expr_to_gbpoly(*eq, &all_vars, pool)?);
    }

    let gb = GroebnerBasis::compute(polys.clone(), MonomialOrder::Lex);
    let gens = gb.generators();

    // Trivial ideal ⟨1⟩ → no solution.
    if gens.len() == 1
        && gens[0].terms.len() == 1
        && gens[0].leading_exp(MonomialOrder::Lex) == Some(vec![0u32; n_vars])
    {
        return Ok(SolutionSet::NoSolution);
    }

    // Candidates are checked against the *input* equations rather than the
    // basis: that is the contract the caller stated, and it does not inherit
    // any mistake the basis computation might have made.
    let finish = |solutions: Vec<Solution>| -> Option<SolutionSet> {
        let had_candidates = !solutions.is_empty();
        let refined = refine_solutions(solutions, &polys, n_vars, pool);
        if had_candidates && refined.is_empty() {
            // Over ℂ a proper ideal always has a zero, so a candidate set that
            // is entirely refuted means the enumeration itself was unsound.
            // Reporting `Finite([])` here would be the worst possible answer —
            // "this system has no solutions" — so decline instead.
            return None;
        }
        Some(SolutionSet::Finite(refined))
    };

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
}
