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
mod rational;
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
                "ensure all equations are polynomial or rational in the declared variables; \
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
    solutions: Vec<Solution>,
    domains: &[GbPoly],
    equations: &[ExprId],
    all_vars: &[ExprId],
    n_solve: usize,
    pool: &ExprPool,
) -> (Vec<Solution>, bool) {
    if domains.is_empty() {
        return (solutions, false);
    }
    let solve_vars: Vec<ExprId> = all_vars[..n_solve].to_vec();
    let mut kept = Vec::with_capacity(solutions.len());
    let mut unproved_drop = false;

    for sol in solutions {
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
            assume_nonzero(h);
        }
        kept.push(sol);
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
/// again by [`exclude_pole_roots`], which decides `D ≠ 0` exactly where it can
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
    // never see one left behind by an earlier solve.  Same for the refusal
    // channel: a stale one would re-label an unrelated `NotPolynomial`.
    let _ = take_solve_side_conditions();
    forget_undefined_equation();
    let n_solve = vars.len();
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
        let cleared = rational::clear_denominators(*eq, &all_vars, pool)?;
        if !cleared.is_polynomial() {
            domains.push(cleared.domain);
        }
        polys.push(cleared.numer);
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
        // Clearing denominators enlarged the solution set; take the excess
        // back out.  An empty list *is* the right answer here — `1/(x−1) =
        // x/(x−1)` genuinely has none — but only when every drop was proved,
        // so a merely-undecided one falls through to the basis instead.
        let (kept, unproved_drop) =
            exclude_pole_roots(refined, &domains, &equations, &all_vars, n_solve, pool);
        if kept.is_empty() && unproved_drop {
            return None;
        }
        Some(SolutionSet::Finite(kept))
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
}
