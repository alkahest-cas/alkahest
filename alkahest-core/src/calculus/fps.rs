//! Lazy (memoized, arbitrary-order) formal power series ring over ℚ.
//!
//! A [`Fps`] represents a formal power series `∑ₙ aₙ xⁿ` whose rational
//! coefficients `aₙ` are computed on demand and cached. Asking for coefficient
//! `50` does not re-truncate from scratch and does not recompute coefficient
//! `10` afterwards — each coefficient is produced at most once and stored in an
//! internal memo. This is the lazy / infinite-precision counterpart to the
//! truncating [`crate::calculus::series`](mod@crate::calculus::series) entry point.
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

impl std::fmt::Debug for Fps<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fps")
            .field("computed", &self.cache.borrow().len())
            .finish_non_exhaustive()
    }
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
    /// by the existing [`crate::calculus::series`](mod@crate::calculus::series) machinery.
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
    /// `… + O(varᵒʳᵈᵉʳ)` format matching [`crate::calculus::series`](mod@crate::calculus::series) output.
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

    // -----------------------------------------------------------------------
    // Analytic operations
    // -----------------------------------------------------------------------

    /// Multiplicative inverse `1/f`, requiring `f(0) ≠ 0`.
    ///
    /// With `b = 1/f`, `b₀ = 1/a₀` and `bₙ = −(1/a₀) ∑_{k=1}^{n} a_k b_{n-k}`.
    pub fn inverse(&self) -> Result<Fps<'p>, FpsError> {
        let a0 = self.coeff(0);
        if a0 == 0 {
            return Err(FpsError::ConstantTermMustBeNonzero);
        }
        let a = self.clone();
        Ok(Fps::from_gen(move |n, prev| {
            if n == 0 {
                Rational::from(1) / a.coeff(0)
            } else {
                let mut acc = Rational::from(0);
                for k in 1..=n {
                    acc += a.coeff(k) * prev[n - k].clone();
                }
                -acc / a.coeff(0)
            }
        }))
    }

    /// Quotient `self / other`, requiring `other(0) ≠ 0`.
    pub fn div(&self, other: &Fps<'p>) -> Result<Fps<'p>, FpsError> {
        Ok(self.mul(&other.inverse()?))
    }

    /// Composition `f ∘ g` where `g(0) = 0`.
    ///
    /// Computed by Horner evaluation of `f` in the series `g`: each coefficient
    /// `n` needs only the partial sum `∑_{k=0}^{n} a_k gᵏ`, and `gᵏ` has
    /// valuation `≥ k`, so coefficient `n` is finite. We accumulate
    /// `∑ a_k gᵏ` by tracking running powers `gᵏ` truncated at the queried
    /// order on demand.
    pub fn compose(&self, g: &Fps<'p>) -> Result<Fps<'p>, FpsError> {
        if g.coeff(0) != 0 {
            return Err(FpsError::ConstantTermMustBeZero);
        }
        let f = self.clone();
        let g = g.clone();
        Ok(Fps::from_fn(move |n| {
            // result coefficient n = ∑_{k=0}^{n} a_k · [xⁿ] gᵏ
            // The k=0 term a₀·g⁰ = a₀ contributes only to coefficient 0.
            if n == 0 {
                return f.coeff(0);
            }
            let mut acc = Rational::from(0);
            // pow holds coefficients of gᵏ for k = current, indices 0..=n.
            let mut pow: Vec<Rational> = vec![Rational::from(0); n + 1];
            pow[0] = Rational::from(1); // g⁰ = 1
            let gc: Vec<Rational> = (0..=n).map(|i| g.coeff(i)).collect();
            for k in 1..=n {
                // pow ← pow * g (Cauchy product), truncated at degree n.
                let mut next = vec![Rational::from(0); n + 1];
                for (i, pi) in pow.iter().enumerate() {
                    if *pi == 0 {
                        continue;
                    }
                    for j in 0..=(n - i) {
                        if gc[j] == 0 {
                            continue;
                        }
                        next[i + j] += pi.clone() * gc[j].clone();
                    }
                }
                pow = next;
                // gᵏ has valuation ≥ k, so [xⁿ] gᵏ = 0 for k > n; loop stops at n.
                acc += f.coeff(k) * pow[n].clone();
            }
            acc
        }))
    }

    /// Compositional inverse (reversion) `h` of `f` with `f(0)=0`, `f'(0)≠0`,
    /// so `f(h(x)) = x` and `h(f(x)) = x`.
    ///
    /// Uses Lagrange inversion: write `f = a₁x·(1 + …)`, set `φ(x) = x/f(x)`
    /// extended to a unit power series; then `[xⁿ] h = (1/n) [x^{n-1}] φ(x)ⁿ`.
    pub fn revert(&self) -> Result<Fps<'p>, FpsError> {
        if self.coeff(0) != 0 {
            return Err(FpsError::ConstantTermMustBeZero);
        }
        if self.coeff(1) == 0 {
            return Err(FpsError::ConstantTermMustBeNonzero);
        }
        // φ(x) = x / f(x) = 1 / (f(x)/x), a unit series (φ(0) = 1/a₁ ≠ 0).
        let f = self.clone();
        // f(x)/x : coefficient n is f_{n+1}.
        let f_over_x = Fps::from_fn(move |n| f.coeff(n + 1));
        let phi = f_over_x.inverse()?; // = x/f(x)
        Ok(Fps::from_fn(move |n| {
            if n == 0 {
                return Rational::from(0);
            }
            // h_n = (1/n) [x^{n-1}] φ(x)ⁿ
            let m = n; // exponent
                       // Compute φⁿ coefficients up to degree n-1.
            let target = n - 1;
            let phic: Vec<Rational> = (0..=target).map(|i| phi.coeff(i)).collect();
            let mut pow: Vec<Rational> = vec![Rational::from(0); target + 1];
            pow[0] = Rational::from(1);
            for _ in 0..m {
                let mut next = vec![Rational::from(0); target + 1];
                for (i, pi) in pow.iter().enumerate() {
                    if *pi == 0 {
                        continue;
                    }
                    for j in 0..=(target - i) {
                        if phic[j] == 0 {
                            continue;
                        }
                        next[i + j] += pi.clone() * phic[j].clone();
                    }
                }
                pow = next;
            }
            pow[target].clone() / Rational::from(n)
        }))
    }

    /// `exp(f)` for a series with `f(0) = 0`.
    ///
    /// With `b = exp(f)`: `b₀ = 1`, and `b' = f' · b` gives the recurrence
    /// `n·bₙ = ∑_{k=1}^{n} k·f_k·b_{n-k}`.
    pub fn exp(&self) -> Result<Fps<'p>, FpsError> {
        if self.coeff(0) != 0 {
            return Err(FpsError::ConstantTermMustBeZero);
        }
        let f = self.clone();
        Ok(Fps::from_gen(move |n, prev| {
            if n == 0 {
                return Rational::from(1);
            }
            let mut acc = Rational::from(0);
            for k in 1..=n {
                acc += Rational::from(k) * f.coeff(k) * prev[n - k].clone();
            }
            acc / Rational::from(n)
        }))
    }

    /// `log(f)` for a series with `f(0) = 1`.
    ///
    /// With `b = log(f)`: `b₀ = 0`, and `b' = f'/f` gives
    /// `n·bₙ = n·f_n − ∑_{k=1}^{n-1} k·b_k·f_{n-k}` (since `f₀ = 1`).
    pub fn log(&self) -> Result<Fps<'p>, FpsError> {
        if self.coeff(0) != 1 {
            return Err(FpsError::ConstantTermMustBeOne);
        }
        let f = self.clone();
        Ok(Fps::from_gen(move |n, prev| {
            if n == 0 {
                return Rational::from(0);
            }
            // n·b_n = n·f_n − ∑_{k=1}^{n-1} k·b_k·f_{n-k}
            let mut acc = Rational::from(n) * f.coeff(n);
            for (k, bk) in prev.iter().enumerate().take(n).skip(1) {
                acc -= Rational::from(k) * bk.clone() * f.coeff(n - k);
            }
            acc / Rational::from(n)
        }))
    }

    /// Binomial power `(1 + f)^α` for rational `α`, requiring `f(0) = 0`.
    ///
    /// Lets `u = 1 + f` (so `u(0) = 1`) and uses `b = uᵅ` with
    /// `u·b' = α·u'·b`, i.e.
    /// `∑ u_j (n−j) b_{n−j} ... ` reduced (with `u₀ = 1`) to
    /// `n·bₙ = ∑_{k=1}^{n} (α(k) − (n−k)) u_k b_{n-k}` where `α(k)=α·k`.
    pub fn pow_binomial(&self, alpha: Rational) -> Result<Fps<'p>, FpsError> {
        if self.coeff(0) != 0 {
            return Err(FpsError::ConstantTermMustBeZero);
        }
        // u = 1 + f
        let f = self.clone();
        let u = Fps::from_fn(move |n| {
            if n == 0 {
                Rational::from(1)
            } else {
                f.coeff(n)
            }
        });
        Ok(Fps::from_gen(move |n, prev| {
            if n == 0 {
                return Rational::from(1);
            }
            // n·b_n = ∑_{k=1}^{n} (α·k − (n−k)) u_k b_{n-k}
            let mut acc = Rational::from(0);
            for k in 1..=n {
                let uk = u.coeff(k);
                if uk == 0 {
                    continue;
                }
                let factor = alpha.clone() * Rational::from(k) - Rational::from(n - k);
                acc += factor * uk * prev[n - k].clone();
            }
            acc / Rational::from(n)
        }))
    }

    /// `m`-th root of a series with `f(0) = 1` (principal branch): `f^{1/m}`.
    pub fn nth_root(&self, m: u32) -> Result<Fps<'p>, FpsError> {
        if m == 0 {
            return Err(FpsError::ConstantTermMustBeOne);
        }
        if self.coeff(0) != 1 {
            return Err(FpsError::ConstantTermMustBeOne);
        }
        // f = 1 + (f - 1); apply binomial with α = 1/m.
        let one = Fps::constant(Rational::from(1));
        let g = self.sub(&one); // g(0) = 0
        g.pow_binomial(Rational::from((1, m as i64)))
    }

    // -----------------------------------------------------------------------
    // Known-series shortcuts (exact rational recurrences)
    // -----------------------------------------------------------------------

    /// `exp(x) = ∑ xⁿ / n!`.
    pub fn exp_series() -> Fps<'static> {
        Fps::from_gen(|n, prev| {
            if n == 0 {
                Rational::from(1)
            } else {
                prev[n - 1].clone() / Rational::from(n)
            }
        })
    }

    /// `log(1 + x) = ∑_{n≥1} (−1)^{n+1} xⁿ / n`.
    pub fn log1p_series() -> Fps<'static> {
        Fps::from_fn(|n| {
            if n == 0 {
                Rational::from(0)
            } else {
                let sign = if n % 2 == 1 { 1 } else { -1 };
                Rational::from((sign, n as i64))
            }
        })
    }

    /// `sin(x) = ∑ (−1)ᵏ x^{2k+1} / (2k+1)!`.
    pub fn sin_series() -> Fps<'static> {
        Fps::from_fn(|n| {
            if n % 2 == 0 {
                Rational::from(0)
            } else {
                let k = (n - 1) / 2;
                let mut fact = Integer::from(1);
                for i in 2..=n {
                    fact *= i as i64;
                }
                let sign = if k % 2 == 0 { 1 } else { -1 };
                Rational::from((Integer::from(sign), fact))
            }
        })
    }

    /// `cos(x) = ∑ (−1)ᵏ x^{2k} / (2k)!`.
    pub fn cos_series() -> Fps<'static> {
        Fps::from_fn(|n| {
            if n % 2 == 1 {
                Rational::from(0)
            } else {
                let k = n / 2;
                let mut fact = Integer::from(1);
                for i in 2..=n {
                    fact *= i as i64;
                }
                let sign = if k % 2 == 0 { 1 } else { -1 };
                Rational::from((Integer::from(sign), fact))
            }
        })
    }

    /// `atan(x) = ∑_{k≥0} (−1)ᵏ x^{2k+1} / (2k+1)`.
    pub fn atan_series() -> Fps<'static> {
        Fps::from_fn(|n| {
            if n % 2 == 0 {
                Rational::from(0)
            } else {
                let k = (n - 1) / 2;
                let sign = if k % 2 == 0 { 1 } else { -1 };
                Rational::from((sign, n as i64))
            }
        })
    }

    /// Binomial series `(1 + x)^α = ∑ C(α, n) xⁿ` for rational `α`.
    ///
    /// `C(α, n) = ∏_{j=0}^{n-1} (α − j) / n!`, via the ratio recurrence
    /// `c_{n} = c_{n-1} · (α − (n−1)) / n`.
    pub fn binomial_series(alpha: Rational) -> Fps<'static> {
        Fps::from_gen(move |n, prev| {
            if n == 0 {
                Rational::from(1)
            } else {
                let prev_c = prev[n - 1].clone();
                prev_c * (alpha.clone() - Rational::from(n - 1)) / Rational::from(n)
            }
        })
    }

    // -----------------------------------------------------------------------
    // Implicit / algebraic series (stretch)
    // -----------------------------------------------------------------------

    /// Lazy series defined *implicitly* by a coefficient functional `step`.
    ///
    /// The series `y` is the unique solution of a fixed point `y = Φ(y)` where
    /// the `n`-th coefficient of `Φ(y)` depends only on the coefficients
    /// `y₀, …, y_{n-1}` (a "well-founded"/contracting recurrence). The caller
    /// supplies `step(n, prev) = yₙ` directly, where `prev = [y₀, …, y_{n-1}]`.
    /// Because each coefficient is produced from strictly earlier ones, the
    /// memoized engine computes the whole series order-by-order with no
    /// truncation.
    ///
    /// This is the building block for algebraic series given by a polynomial
    /// equation. For example the Catalan generating function
    /// `C(x) = 1 + x·C(x)²` (with `C₀ = 1`) has, for `n ≥ 1`,
    /// `Cₙ = [x^{n-1}] C² = ∑_{k=0}^{n-1} C_k C_{n-1-k}`, which references only
    /// `C₀, …, C_{n-1}`:
    ///
    /// ```
    /// use alkahest_cas::calculus::fps::Fps;
    /// use rug::Rational;
    /// let catalan = Fps::implicit(|n, prev| {
    ///     if n == 0 {
    ///         Rational::from(1)
    ///     } else {
    ///         // coefficient n-1 of C², using prev = [C_0..C_{n-1}]
    ///         let mut acc = Rational::from(0);
    ///         for k in 0..n {
    ///             acc += prev[k].clone() * prev[n - 1 - k].clone();
    ///         }
    ///         acc
    ///     }
    /// });
    /// let expected = [1, 1, 2, 5, 14, 42, 132, 429];
    /// for (n, &e) in expected.iter().enumerate() {
    ///     assert_eq!(catalan.coeff(n), Rational::from(e));
    /// }
    /// ```
    pub fn implicit<F>(step: F) -> Self
    where
        F: Fn(usize, &[Rational]) -> Rational + 'p,
    {
        Fps::from_gen(step)
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
        // Coefficients from `local_expansion` may still carry unevaluated
        // numeric heads (e.g. `exp(0)`); simplify before extracting the value.
        let cs = crate::simplify::simplify(c, pool).value;
        let r = expr_to_rational(cs, pool).ok_or(FpsError::NonRationalCoefficient)?;
        out[idx] = r;
    }
    Ok(out)
}

/// Strictly fold a (simplified) numeric expression into an exact rational,
/// returning `None` for anything that is not a closed rational number.
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
            ExprData::Integer(ref n) => {
                let ei = n.0.to_i32()?;
                let b = expr_to_rational(base, pool)?;
                if ei == 0 {
                    Some(Rational::from(1))
                } else if ei > 0 {
                    let mut acc = Rational::from(1);
                    for _ in 0..ei {
                        acc *= b.clone();
                    }
                    Some(acc)
                } else {
                    if b == 0 {
                        return None;
                    }
                    let mut acc = Rational::from(1);
                    for _ in 0..(-ei) {
                        acc *= b.clone();
                    }
                    Some(Rational::from(1) / acc)
                }
            }
            _ => None,
        },
        // Transcendental heads that occur in series-at-0 coefficients with an
        // exact rational value when the argument folds to a known point (e.g.
        // `exp(0) = 1`, `cos(0) = 1`, `sin(0) = 0`, `log(1) = 0`). `simplify`
        // does not always collapse these, so fold them here.
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let a = expr_to_rational(args[0], pool)?;
            let name = name.as_str();
            if a == 0 {
                match name {
                    "exp" | "cos" | "cosh" => Some(Rational::from(1)),
                    "sin" | "sinh" | "tan" | "tanh" | "atan" | "asin" | "asinh" => {
                        Some(Rational::from(0))
                    }
                    _ => None,
                }
            } else if a == 1 && (name == "log" || name == "ln") {
                Some(Rational::from(0))
            } else {
                None
            }
        }
        _ => None,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use rug::ops::Pow;

    fn r(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }
    fn ri(n: i64) -> Rational {
        Rational::from(n)
    }
    fn factorial(n: u64) -> Integer {
        let mut f = Integer::from(1);
        for i in 2..=n {
            f *= i;
        }
        f
    }

    #[test]
    fn exp_coeffs_match_one_over_factorial() {
        let e = Fps::exp_series();
        for n in 0..15u64 {
            assert_eq!(
                e.coeff(n as usize),
                Rational::from((Integer::from(1), factorial(n)))
            );
        }
    }

    #[test]
    fn log1p_coeffs() {
        let l = Fps::log1p_series();
        assert_eq!(l.coeff(0), ri(0));
        assert_eq!(l.coeff(1), ri(1));
        assert_eq!(l.coeff(2), r(-1, 2));
        assert_eq!(l.coeff(3), r(1, 3));
        assert_eq!(l.coeff(4), r(-1, 4));
    }

    #[test]
    fn sin_cos_coeffs() {
        let s = Fps::sin_series();
        assert_eq!(s.coeff(1), ri(1));
        assert_eq!(s.coeff(3), r(-1, 6));
        assert_eq!(s.coeff(5), r(1, 120));
        assert_eq!(s.coeff(2), ri(0));
        let c = Fps::cos_series();
        assert_eq!(c.coeff(0), ri(1));
        assert_eq!(c.coeff(2), r(-1, 2));
        assert_eq!(c.coeff(4), r(1, 24));
        assert_eq!(c.coeff(1), ri(0));
    }

    #[test]
    fn binomial_series_half() {
        // (1+x)^{1/2}: 1, 1/2, -1/8, 1/16, -5/128, …
        let b = Fps::binomial_series(r(1, 2));
        assert_eq!(b.coeff(0), ri(1));
        assert_eq!(b.coeff(1), r(1, 2));
        assert_eq!(b.coeff(2), r(-1, 8));
        assert_eq!(b.coeff(3), r(1, 16));
        assert_eq!(b.coeff(4), r(-5, 128));
    }

    #[test]
    fn mul_consistency_with_series_truncation() {
        // sin·cos should equal (1/2) sin(2x); check via Cauchy product coeffs.
        let s = Fps::sin_series();
        let c = Fps::cos_series();
        let prod = s.mul(&c);
        // (1/2) sin(2x) = ∑ (-1)^k 2^{2k} x^{2k+1} / (2k+1)!
        for k in 0..6u64 {
            let n = 2 * k + 1;
            let sign = if k % 2 == 0 { 1 } else { -1 };
            let two_pow = Integer::from(2).pow((2 * k) as u32);
            let expected = Rational::from((Integer::from(sign) * two_pow, factorial(n)));
            assert_eq!(prod.coeff(n as usize), expected);
        }
        // even coefficients vanish
        for k in 0..6 {
            assert_eq!(prod.coeff(2 * k), ri(0));
        }
    }

    #[test]
    fn derivative_and_integral_inverse() {
        let e = Fps::exp_series();
        // d/dx exp = exp
        let de = e.derivative();
        for n in 0..12 {
            assert_eq!(de.coeff(n), e.coeff(n));
        }
        // ∫ exp from 0 = exp - 1
        let ie = e.integral();
        assert_eq!(ie.coeff(0), ri(0));
        for n in 1..12 {
            assert_eq!(ie.coeff(n), e.coeff(n));
        }
    }

    #[test]
    fn multiplicative_inverse() {
        // 1/(1-x) = geometric series 1,1,1,…
        let one_minus_x = Fps::from_poly(&[ri(1), ri(-1)]);
        let inv = one_minus_x.inverse().unwrap();
        for n in 0..20 {
            assert_eq!(inv.coeff(n), ri(1));
        }
        // f · (1/f) = 1
        let e = Fps::exp_series();
        let prod = e.mul(&e.inverse().unwrap());
        assert_eq!(prod.coeff(0), ri(1));
        for n in 1..15 {
            assert_eq!(prod.coeff(n), ri(0));
        }
    }

    #[test]
    fn exp_log_roundtrip_to_order_30() {
        // exp(log(1+x)) = 1 + x exactly to order 30.
        let l = Fps::log1p_series(); // l(0) = 0
        let e = l.exp().unwrap();
        assert_eq!(e.coeff(0), ri(1));
        assert_eq!(e.coeff(1), ri(1));
        for n in 2..=30 {
            assert_eq!(e.coeff(n), ri(0), "coeff {n} should vanish");
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        // log(exp(x)) = x.
        let e = Fps::exp_series(); // e(0) = 1
        let l = e.log().unwrap();
        assert_eq!(l.coeff(0), ri(0));
        assert_eq!(l.coeff(1), ri(1));
        for n in 2..=20 {
            assert_eq!(l.coeff(n), ri(0));
        }
    }

    #[test]
    fn reversion_of_sin_is_arcsin() {
        // arcsin coefficients: x + x³/6 + 3x⁵/40 + 15 x⁷/336 + …
        let s = Fps::sin_series();
        let asin = s.revert().unwrap();
        assert_eq!(asin.coeff(1), ri(1));
        assert_eq!(asin.coeff(3), r(1, 6));
        assert_eq!(asin.coeff(5), r(3, 40));
        assert_eq!(asin.coeff(7), r(15, 336));
        assert_eq!(asin.coeff(2), ri(0));
        assert_eq!(asin.coeff(4), ri(0));
    }

    #[test]
    fn reversion_roundtrip() {
        // revert(revert(f)) = f for f = sin.
        let s = Fps::sin_series();
        let rr = s.revert().unwrap().revert().unwrap();
        for n in 0..10 {
            assert_eq!(rr.coeff(n), s.coeff(n));
        }
    }

    #[test]
    fn composition_consistency() {
        // exp(sin(x)) compose vs exp_series ∘ sin_series; check low coeffs.
        // exp(sin x) = 1 + x + x²/2 - x⁴/8 - x⁵/15 + …
        let e = Fps::exp_series();
        let s = Fps::sin_series();
        let comp = e.compose(&s).unwrap();
        assert_eq!(comp.coeff(0), ri(1));
        assert_eq!(comp.coeff(1), ri(1));
        assert_eq!(comp.coeff(2), r(1, 2));
        assert_eq!(comp.coeff(3), ri(0));
        assert_eq!(comp.coeff(4), r(-1, 8));
        assert_eq!(comp.coeff(5), r(-1, 15));
    }

    #[test]
    fn catalan_numbers_via_binomial_half() {
        // C(x) = (1 - sqrt(1-4x)) / (2x); coefficients are Catalan numbers.
        // sqrt(1-4x) = (1 + (-4x))^{1/2}: substitute -4x into (1+u)^{1/2}.
        let sqrt_1m4x = Fps::from_fn(|n| {
            // coefficient of x^n in (1-4x)^{1/2} = C(1/2, n) · (-4)^n
            let mut binom = ri(1);
            for j in 0..n {
                binom *= r(1, 2) - ri(j as i64);
            }
            binom /= Rational::from(factorial(n as u64));
            binom * Rational::from(Integer::from(-4).pow(n as u32))
        });
        // numerator = 1 - sqrt(1-4x); divide by 2x → shift down by 1, scale 1/2
        let one = Fps::constant(ri(1));
        let num = one.sub(&sqrt_1m4x); // valuation 1 (num[0] = 0)
        let catalan = Fps::from_fn(move |n| num.coeff(n + 1) / ri(2));
        let expected = [1i64, 1, 2, 5, 14, 42, 132, 429];
        for (n, &e) in expected.iter().enumerate() {
            assert_eq!(catalan.coeff(n), ri(e), "Catalan C_{n}");
        }
    }

    #[test]
    fn catalan_via_implicit_equation() {
        // C(x) = 1 + x·C(x)² solved directly as an algebraic/implicit series.
        let catalan = Fps::implicit(|n, prev| {
            if n == 0 {
                ri(1)
            } else {
                let mut acc = ri(0);
                for k in 0..n {
                    acc += prev[k].clone() * prev[n - 1 - k].clone();
                }
                acc
            }
        });
        let expected = [1i64, 1, 2, 5, 14, 42, 132, 429, 1430];
        for (n, &e) in expected.iter().enumerate() {
            assert_eq!(catalan.coeff(n), ri(e), "Catalan C_{n}");
        }
        // Cross-check against the binomial closed form computed above.
        let sqrt_1m4x = Fps::from_fn(|n| {
            let mut binom = ri(1);
            for j in 0..n {
                binom *= r(1, 2) - ri(j as i64);
            }
            binom /= Rational::from(factorial(n as u64));
            binom * Rational::from(Integer::from(-4).pow(n as u32))
        });
        let one = Fps::constant(ri(1));
        let num = one.sub(&sqrt_1m4x);
        let closed = Fps::from_fn(move |n| num.coeff(n + 1) / ri(2));
        for n in 0..12 {
            assert_eq!(catalan.coeff(n), closed.coeff(n));
        }
    }

    #[test]
    fn nth_root_squares_back() {
        // (sqrt(1+x))² = 1+x.
        let one_plus_x = Fps::from_poly(&[ri(1), ri(1)]);
        let root = one_plus_x.nth_root(2).unwrap();
        let sq = root.mul(&root);
        assert_eq!(sq.coeff(0), ri(1));
        assert_eq!(sq.coeff(1), ri(1));
        for n in 2..15 {
            assert_eq!(sq.coeff(n), ri(0));
        }
    }

    #[test]
    fn pow_binomial_matches_binomial_series() {
        // (1+x)^α via pow_binomial(f=x) == binomial_series(α).
        let x = Fps::x();
        let alpha = r(2, 3);
        let p = x.pow_binomial(alpha.clone()).unwrap();
        let b = Fps::binomial_series(alpha);
        for n in 0..12 {
            assert_eq!(p.coeff(n), b.coeff(n));
        }
    }

    #[test]
    fn laziness_high_then_low() {
        // Request coeff(40) first, then coeff(10): both correct, no recompute issue.
        let e = Fps::exp_series();
        let c40 = e.coeff(40);
        assert_eq!(c40, Rational::from((Integer::from(1), factorial(40))));
        let c10 = e.coeff(10);
        assert_eq!(c10, Rational::from((Integer::from(1), factorial(10))));
        // cache should hold at least 41 entries
        assert!(e.cache.borrow().len() >= 41);
    }

    #[test]
    fn from_rational_geometric() {
        // 1/(1 - x - x²) = Fibonacci generating function: 1,1,2,3,5,8,13,…
        let f = Fps::from_rational(&[ri(1)], &[ri(1), ri(-1), ri(-1)]).unwrap();
        let fib = [1i64, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (n, &v) in fib.iter().enumerate() {
            assert_eq!(f.coeff(n), ri(v));
        }
    }

    #[test]
    fn from_rational_rejects_zero_denominator() {
        let e = Fps::from_rational(&[ri(1)], &[ri(0), ri(1)]);
        assert_eq!(e.unwrap_err(), FpsError::DenominatorVanishesAtZero);
    }

    #[test]
    fn from_expr_matches_series() {
        // expr-backed series of exp(x) matches exp_series coefficients.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let ex = pool.func("exp", vec![x]);
        let fps = Fps::from_expr(ex, x, &pool).unwrap();
        let known = Fps::exp_series();
        for n in 0..12 {
            assert_eq!(fps.coeff(n), known.coeff(n), "coeff {n}");
        }
        // laziness across the series machinery: high then low.
        let c20 = fps.coeff(20);
        assert_eq!(c20, known.coeff(20));
        assert_eq!(fps.coeff(5), known.coeff(5));
    }

    #[test]
    fn from_expr_rejects_polar() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let inv_x = pool.pow(x, pool.integer(-1));
        let e = Fps::from_expr(inv_x, x, &pool);
        assert_eq!(e.unwrap_err(), FpsError::NotAnalyticAtZero);
    }

    #[test]
    fn to_expr_format() {
        // truncate exp to order 4 → 1 + x + x²/2 + x³/6 + O(x⁴)
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = Fps::exp_series();
        let expr = e.to_expr(x, 4, &pool);
        // The result must contain a BigO term.
        fn has_big_o(id: ExprId, pool: &ExprPool) -> bool {
            match pool.get(id) {
                ExprData::BigO(_) => true,
                ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|e| has_big_o(*e, pool)),
                ExprData::Pow { base, exp } => has_big_o(base, pool) || has_big_o(exp, pool),
                _ => false,
            }
        }
        assert!(has_big_o(expr, &pool));
    }

    #[test]
    fn inverse_requires_nonzero_constant() {
        let f = Fps::x(); // f(0) = 0
        assert_eq!(
            f.inverse().unwrap_err(),
            FpsError::ConstantTermMustBeNonzero
        );
    }

    #[test]
    fn exp_requires_zero_constant() {
        let f = Fps::constant(ri(1));
        assert_eq!(f.exp().unwrap_err(), FpsError::ConstantTermMustBeZero);
    }

    #[test]
    fn log_requires_unit_constant() {
        let f = Fps::x();
        assert_eq!(f.log().unwrap_err(), FpsError::ConstantTermMustBeOne);
    }
}
