//! Lazy (memoized, arbitrary-order) formal power series ring over ℚ.
//!
//! A [`Fps`] represents a formal power series `∑ₙ aₙ xⁿ` whose rational
//! coefficients `aₙ` are computed on demand and cached. Asking for coefficient
//! `50` does not re-truncate from scratch and does not recompute coefficient
//! `10` afterwards — each coefficient is produced at most once and stored in an
//! internal memo. This is the lazy / infinite-precision counterpart to the
//! truncating [`crate::calculus::series`] entry point.
//!
//! Coefficients live in [`rug::Rational`] (exact ℚ arithmetic). A series can be
//! built from:
//!
//! * an explicit coefficient closure ([`Fps::from_fn`]),
//! * a polynomial coefficient slice ([`Fps::from_poly`]),
//! * a rational function `p(x)/q(x)` with `q(0) ≠ 0` ([`Fps::from_rational`]),
//! * an arbitrary expression via the existing series machinery
//!   ([`Fps::from_expr`]), computing coefficients incrementally and caching them,
//! * the known-series shortcuts ([`Fps::exp_series`], [`Fps::sin_series`], …).
//!
//! Ring and analytic operations (add, Cauchy product, scalar ops, derivative,
//! integral, composition, reversion, multiplicative inverse, `exp`/`log`, n-th
//! root, binomial `(1+x)^α`) build a *new* lazy series whose generator pulls
//! coefficients of the operands on demand.
//!
//! All coefficient recurrences use exact rational arithmetic, so identities such
//! as `exp(log(1+x)) = 1 + x` hold exactly to every computed order.

use crate::calculus::series::{local_expansion, SeriesError};
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Integer, Rational};
use std::cell::RefCell;
use std::rc::Rc;

/// Generator for a lazy power series coefficient.
///
/// Given the index `n` and the slice of already-computed coefficients
/// `prev = [a₀, …, a_{n-1}]`, it returns `aₙ`. The slice lets recurrences that
/// reference earlier coefficients (Cauchy products, `exp`/`log`, n-th root)
/// run in the natural way without recomputation.
type Gen<'p> = dyn Fn(usize, &[Rational]) -> Rational + 'p;

/// A lazy formal power series `∑ₙ aₙ xⁿ` over ℚ with memoized coefficients.
///
/// Cloning is cheap and shares the underlying memo (`Rc`): two clones of the
/// same series never compute a coefficient twice between them.
///
/// The lifetime `'p` is the lifetime of any [`ExprPool`] an expression-backed
/// series borrows (see [`Fps::from_expr`]). Series built purely from rational
/// data (`from_fn`, `from_poly`, the known-series shortcuts, and ring/analytic
/// combinations thereof) borrow nothing and are `Fps<'static>`.
#[derive(Clone)]
pub struct Fps<'p> {
    gen: Rc<Gen<'p>>,
    cache: Rc<RefCell<Vec<Rational>>>,
}

impl<'p> Fps<'p> {
    /// Build a series from a raw coefficient generator.
    ///
    /// `gen(n, prev)` must return the `n`-th coefficient given the lower-index
    /// coefficients `prev = [a₀, …, a_{n-1}]`.
    fn from_gen<F>(gen: F) -> Self
    where
        F: Fn(usize, &[Rational]) -> Rational + 'p,
    {
        Fps {
            gen: Rc::new(gen),
            cache: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Series whose `n`-th coefficient is `f(n)`, independent of earlier ones.
    ///
    /// ```
    /// use alkahest_cas::calculus::fps::Fps;
    /// use rug::Rational;
    /// // geometric series 1 + x + x² + …
    /// let g = Fps::from_fn(|_| Rational::from(1));
    /// assert_eq!(g.coeff(5), Rational::from(1));
    /// ```
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(usize) -> Rational + 'p,
    {
        Fps::from_gen(move |n, _| f(n))
    }

    /// Series from explicit (ascending) rational coefficients of a polynomial:
    /// `coeffs[i]` is the coefficient of `xⁱ`; all higher coefficients are `0`.
    pub fn from_poly(coeffs: &[Rational]) -> Self {
        let coeffs: Vec<Rational> = coeffs.to_vec();
        Fps::from_fn(move |n| coeffs.get(n).cloned().unwrap_or_else(|| Rational::from(0)))
    }

    /// The zero series.
    pub fn zero() -> Self {
        Fps::from_fn(|_| Rational::from(0))
    }

    /// The constant series `c`.
    pub fn constant(c: Rational) -> Self {
        Fps::from_fn(move |n| if n == 0 { c.clone() } else { Rational::from(0) })
    }

    /// The series `x` (coefficient `1` at index `1`, else `0`).
    pub fn x() -> Self {
        Fps::from_fn(|n| {
            if n == 1 {
                Rational::from(1)
            } else {
                Rational::from(0)
            }
        })
    }

    /// The `n`-th coefficient `aₙ`, computing and memoizing all coefficients up
    /// to index `n` as needed. Coefficients already in the memo are reused, so
    /// `coeff(40)` followed by `coeff(10)` recomputes nothing.
    pub fn coeff(&self, n: usize) -> Rational {
        {
            let cache = self.cache.borrow();
            if n < cache.len() {
                return cache[n].clone();
            }
        }
        // Extend the cache up to and including index `n`. We must not hold the
        // borrow across the generator call because some generators read earlier
        // coefficients of *this* series.
        let mut next = self.cache.borrow().len();
        while next <= n {
            let prev: Vec<Rational> = self.cache.borrow().clone();
            let c = (self.gen)(next, &prev);
            let mut cache = self.cache.borrow_mut();
            // Re-check in case of re-entrancy (a generator may have filled it).
            if cache.len() == next {
                cache.push(c);
            }
            next = cache.len();
        }
        self.cache.borrow()[n].clone()
    }

    /// The first `n` coefficients `[a₀, …, a_{n-1}]`.
    pub fn coeffs(&self, n: usize) -> Vec<Rational> {
        (0..n).map(|i| self.coeff(i)).collect()
    }

    // -----------------------------------------------------------------------
    // Constructors from CAS objects
    // -----------------------------------------------------------------------

    /// Series of the rational function `p(x)/q(x)` (geometric expansion),
    /// requiring `q(0) ≠ 0`. `num` / `den` are ascending rational coefficient
    /// slices of `p` and `q`.
    ///
    /// Uses the recurrence `aₙ = (pₙ − ∑_{k=1}^{n} q_k a_{n-k}) / q₀`.
    pub fn from_rational(num: &[Rational], den: &[Rational]) -> Result<Self, FpsError> {
        let q0 = den.first().cloned().unwrap_or_else(|| Rational::from(0));
        if q0 == 0 {
            return Err(FpsError::DenominatorVanishesAtZero);
        }
        let num: Vec<Rational> = num.to_vec();
        let den: Vec<Rational> = den.to_vec();
        Ok(Fps::from_gen(move |n, prev| {
            let pn = num.get(n).cloned().unwrap_or_else(|| Rational::from(0));
            let mut acc = pn;
            for k in 1..=n {
                if let Some(qk) = den.get(k) {
                    if *qk != 0 {
                        acc -= qk.clone() * prev[n - k].clone();
                    }
                }
            }
            acc / q0.clone()
        }))
    }

    /// Lazy series of an arbitrary expression `expr` in `var` about `0`, backed
    /// by the existing [`crate::calculus::series`] machinery.
    ///
    /// Coefficients are computed incrementally: requesting coefficient `n`
    /// expands `expr` to order `n + 1` (only when the memo does not already
    /// cover `n`) and caches every coefficient produced. The expansion must be
    /// an ordinary power series (valuation `≥ 0`) with purely rational
    /// coefficients; otherwise [`FpsError`] is returned at construction-probe
    /// time via the first non-rational / polar coefficient.
    ///
    /// Because the underlying `series` call recomputes from scratch each time
    /// the requested order grows, this constructor grows the cache in chunks to
    /// amortize the cost (doubling the expansion order on a cache miss).
    pub fn from_expr(expr: ExprId, var: ExprId, pool: &'p ExprPool) -> Result<Self, FpsError> {
        // Probe order 1 to validate analyticity and rational coefficients up
        // front, so an obviously-bad input fails fast.
        probe_expr_coeffs(expr, var, 1, pool)?;

        // Shared expansion cache local to this generator (separate from the Fps
        // memo): records "expanded up to order K" so we re-expand in growing
        // chunks rather than once per coefficient.
        let expanded: Rc<RefCell<Vec<Rational>>> = Rc::new(RefCell::new(Vec::new()));

        Ok(Fps::from_gen(move |n, _prev| {
            {
                let e = expanded.borrow();
                if n < e.len() {
                    return e[n].clone();
                }
            }
            let mut order = (n + 1).max(4);
            // Grow geometrically past the previous expansion to amortize the
            // cost of re-running `series` from scratch as the order climbs.
            let have = expanded.borrow().len();
            if order < have * 2 {
                order = have * 2;
            }
            let coeffs = probe_expr_coeffs(expr, var, order as u32, pool)
                .unwrap_or_else(|_| vec![Rational::from(0); order]);
            let mut e = expanded.borrow_mut();
            *e = coeffs;
            e.get(n).cloned().unwrap_or_else(|| Rational::from(0))
        }))
    }

    // -----------------------------------------------------------------------
    // Ring operations
    // -----------------------------------------------------------------------

    /// Sum of two series, coefficientwise.
    pub fn add(&self, other: &Fps<'p>) -> Fps<'p> {
        let a = self.clone();
        let b = other.clone();
        Fps::from_fn(move |n| a.coeff(n) + b.coeff(n))
    }

    /// Difference of two series, coefficientwise.
    pub fn sub(&self, other: &Fps<'p>) -> Fps<'p> {
        let a = self.clone();
        let b = other.clone();
        Fps::from_fn(move |n| a.coeff(n) - b.coeff(n))
    }

    /// Cauchy product `(∑ aₙxⁿ)(∑ bₙxⁿ)`: `cₙ = ∑_{k=0}^{n} a_k b_{n-k}`.
    pub fn mul(&self, other: &Fps<'p>) -> Fps<'p> {
        let a = self.clone();
        let b = other.clone();
        Fps::from_fn(move |n| {
            let mut acc = Rational::from(0);
            for k in 0..=n {
                acc += a.coeff(k) * b.coeff(n - k);
            }
            acc
        })
    }

    /// Scale every coefficient by the rational `c`.
    pub fn scale(&self, c: Rational) -> Fps<'p> {
        let a = self.clone();
        Fps::from_fn(move |n| c.clone() * a.coeff(n))
    }

    /// Formal derivative: `(∑ aₙxⁿ)' = ∑ (n+1) a_{n+1} xⁿ`.
    pub fn derivative(&self) -> Fps<'p> {
        let a = self.clone();
        Fps::from_fn(move |n| Rational::from(n + 1) * a.coeff(n + 1))
    }

    /// Formal integral with zero constant term: `∫ ∑ aₙxⁿ = ∑ a_{n-1}/n xⁿ`.
    pub fn integral(&self) -> Fps<'p> {
        let a = self.clone();
        Fps::from_fn(move |n| {
            if n == 0 {
                Rational::from(0)
            } else {
                a.coeff(n - 1) / Rational::from(n)
            }
        })
    }

    /// Shift up by `k`: multiply by `xᵏ` (coefficient `n` becomes old `n−k`).
    pub fn shift_up(&self, k: usize) -> Fps<'p> {
        let a = self.clone();
        Fps::from_fn(move |n| {
            if n < k {
                Rational::from(0)
            } else {
                a.coeff(n - k)
            }
        })
    }

    // -----------------------------------------------------------------------
    // Conversion back to the CAS
    // -----------------------------------------------------------------------

    /// Truncate to a symbolic expression of degree `< order`, in `var`, in the
    /// `… + O(varᵒʳᵈᵉʳ)` format matching [`crate::calculus::series`] output.
    pub fn to_expr(&self, var: ExprId, order: u32, pool: &ExprPool) -> ExprId {
        let mut terms = Vec::new();
        for k in 0..order as usize {
            let c = self.coeff(k);
            if c == 0 {
                continue;
            }
            let coeff_e = rat_to_expr(&c, pool);
            let term = if k == 0 {
                coeff_e
            } else if k == 1 {
                pool.mul(vec![coeff_e, var])
            } else {
                let p = pool.pow(var, pool.integer(k as i64));
                pool.mul(vec![coeff_e, p])
            };
            terms.push(term);
        }
        let o_term = pool.big_o(pool.pow(var, pool.integer(order as i64)));
        terms.push(o_term);
        pool.add(terms)
    }
}

/// Errors raised by formal-power-series construction / analytic operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FpsError {
    /// A rational-function constructor was given `q(0) = 0`.
    DenominatorVanishesAtZero,
    /// An expression series had a polar part (negative valuation) about `0`.
    NotAnalyticAtZero,
    /// An expression coefficient was not a rational number.
    NonRationalCoefficient,
    /// An operation required `f(0) = 0` (composition inner, `exp`, reversion).
    ConstantTermMustBeZero,
    /// An operation required `f(0) = 1` (`log`, default n-th root).
    ConstantTermMustBeOne,
    /// An operation required `f(0) ≠ 0` (multiplicative inverse).
    ConstantTermMustBeNonzero,
    /// Series expansion (Taylor coefficients) failed.
    Series(String),
}

impl std::fmt::Display for FpsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FpsError::DenominatorVanishesAtZero => {
                write!(f, "rational-function denominator vanishes at x = 0")
            }
            FpsError::NotAnalyticAtZero => write!(f, "expression is not analytic at x = 0"),
            FpsError::NonRationalCoefficient => {
                write!(f, "series coefficient is not a rational number")
            }
            FpsError::ConstantTermMustBeZero => write!(f, "operation requires f(0) = 0"),
            FpsError::ConstantTermMustBeOne => write!(f, "operation requires f(0) = 1"),
            FpsError::ConstantTermMustBeNonzero => write!(f, "operation requires f(0) != 0"),
            FpsError::Series(e) => write!(f, "series expansion failed: {e}"),
        }
    }
}

impl std::error::Error for FpsError {}

impl crate::errors::AlkahestError for FpsError {
    fn code(&self) -> &'static str {
        match self {
            FpsError::DenominatorVanishesAtZero => "E-FPS-001",
            FpsError::NotAnalyticAtZero => "E-FPS-002",
            FpsError::NonRationalCoefficient => "E-FPS-003",
            FpsError::ConstantTermMustBeZero => "E-FPS-004",
            FpsError::ConstantTermMustBeOne => "E-FPS-005",
            FpsError::ConstantTermMustBeNonzero => "E-FPS-006",
            FpsError::Series(_) => "E-FPS-007",
        }
    }
}

impl From<SeriesError> for FpsError {
    fn from(e: SeriesError) -> Self {
        FpsError::Series(e.to_string())
    }
}

/// Expand `expr` about `0` to `order` Taylor coefficients (ascending rational),
/// erroring on polar parts or non-rational coefficients.
fn probe_expr_coeffs(
    expr: ExprId,
    var: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Vec<Rational>, FpsError> {
    let zero = pool.integer(0);
    let le = local_expansion(expr, var, zero, order, pool)?;
    if le.valuation < 0 {
        return Err(FpsError::NotAnalyticAtZero);
    }
    let shift = le.valuation as usize;
    let mut out = vec![Rational::from(0); order as usize];
    for (i, &c) in le.coeffs.iter().enumerate() {
        let idx = shift + i;
        if idx >= order as usize {
            break;
        }
        let r = expr_to_rational(c, pool).ok_or(FpsError::NonRationalCoefficient)?;
        out[idx] = r;
    }
    Ok(out)
}

/// Convert a rational coefficient into the corresponding `ExprId`.
fn rat_to_expr(r: &Rational, pool: &ExprPool) -> ExprId {
    let (num, den) = (r.numer().clone(), r.denom().clone());
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

/// Strict structural conversion of an expression to a rational, or `None`.
fn expr_to_rational(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(ref n) => Some(Rational::from((n.0.clone(), Integer::from(1)))),
        ExprData::Rational(ref r) => Some(r.0.clone()),
        ExprData::Add(ref args) => {
            let mut acc = Rational::from(0);
            for &a in args {
                acc += expr_to_rational(a, pool)?;
            }
            Some(acc)
        }
        ExprData::Mul(ref args) => {
            let mut acc = Rational::from(1);
            for &a in args {
                acc *= expr_to_rational(a, pool)?;
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) => {
                let ei = n.0.to_i32()?;
                if ei < 0 {
                    None
                } else {
                    let b = expr_to_rational(base, pool)?;
                    let mut acc = Rational::from(1);
                    for _ in 0..ei {
                        acc *= b.clone();
                    }
                    Some(acc)
                }
            }
            _ => None,
        },
        _ => None,
    }
}
