//! Exact sparse arithmetic in `Q[n, j, k]` and `Q(n, j, k)`.
//!
//! The single-sum engine ([`super::super::qfield`]) builds a careful tower
//! `Q(n)[k]` / `Q(n)(k)` with a subresultant-PRS gcd, because Gosper's
//! algorithm needs a genuine normal form. The double-sum ansatz search in
//! [`super::search`] never calls a gcd: it fixes the certificate
//! denominators up front from the term's own shift-ratio denominators (see
//! that module's docs) and solves a *linear* system for the numerators. So
//! this module is deliberately simpler than `qfield` — a plain sparse
//! trivariate polynomial ring over `Q`, dense enough operations
//! (add/mul/shift/substitute) and nothing else. `Rat3` (the field of
//! fractions) is kept unreduced throughout; the only place that matters is
//! the final exact identity check, which is a zero-test on a cross-multiplied
//! numerator, not a canonical form.

use rug::ops::Pow as _;
use rug::Rational;
use std::collections::BTreeMap;

/// Which of the three variables `(n, j, k)` an operation acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    N,
    J,
    K,
}

/// Exponents `(deg_n, deg_j, deg_k)` of one monomial.
pub type Exp = (u32, u32, u32);

fn exp_axis(e: Exp, axis: Axis) -> u32 {
    match axis {
        Axis::N => e.0,
        Axis::J => e.1,
        Axis::K => e.2,
    }
}

fn exp_with_axis(mut e: Exp, axis: Axis, v: u32) -> Exp {
    match axis {
        Axis::N => e.0 = v,
        Axis::J => e.1 = v,
        Axis::K => e.2 = v,
    }
    e
}

fn exp_add(a: Exp, b: Exp) -> Exp {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

/// A polynomial in `Q[n, j, k]`, stored as a sparse map from exponent triples
/// to non-zero rational coefficients.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Poly3 {
    pub terms: BTreeMap<Exp, Rational>,
}

impl Poly3 {
    pub fn zero() -> Self {
        Poly3 {
            terms: BTreeMap::new(),
        }
    }

    pub fn one() -> Self {
        Poly3::constant(Rational::from(1))
    }

    pub fn constant(q: Rational) -> Self {
        let mut terms = BTreeMap::new();
        if q != 0 {
            terms.insert((0, 0, 0), q);
        }
        Poly3 { terms }
    }

    pub fn from_i64(v: i64) -> Self {
        Poly3::constant(Rational::from(v))
    }

    pub fn var(axis: Axis) -> Self {
        let e = exp_with_axis((0, 0, 0), axis, 1);
        let mut terms = BTreeMap::new();
        terms.insert(e, Rational::from(1));
        Poly3 { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.terms.len() == 1 && self.terms.get(&(0, 0, 0)).map(|c| *c == 1).unwrap_or(false)
    }

    /// The as-constructed constant value, or `None` if `self` is not
    /// (syntactically) a constant.
    pub fn as_constant(&self) -> Option<Rational> {
        if self.terms.is_empty() {
            return Some(Rational::from(0));
        }
        if self.terms.len() == 1 {
            if let Some(c) = self.terms.get(&(0, 0, 0)) {
                return Some(c.clone());
            }
        }
        None
    }

    /// Highest exponent on `axis` among the non-zero terms, or `-1` for the
    /// zero polynomial.
    ///
    /// Not called outside this module's own tests today — kept as a general
    /// primitive (and exercised by them) rather than deleted, since it is
    /// the natural thing a future extension of the search or boundary
    /// analysis would reach for.
    #[allow(dead_code)]
    pub fn degree(&self, axis: Axis) -> i64 {
        self.terms
            .keys()
            .map(|e| exp_axis(*e, axis) as i64)
            .max()
            .unwrap_or(-1)
    }

    pub fn neg(&self) -> Poly3 {
        Poly3 {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (*e, Rational::from(-c)))
                .collect(),
        }
    }

    pub fn add(&self, other: &Poly3) -> Poly3 {
        let mut out = self.terms.clone();
        for (e, c) in &other.terms {
            let entry = out.entry(*e).or_insert_with(|| Rational::from(0));
            *entry += c;
        }
        out.retain(|_, c| *c != 0);
        Poly3 { terms: out }
    }

    pub fn sub(&self, other: &Poly3) -> Poly3 {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Poly3) -> Poly3 {
        if self.is_zero() || other.is_zero() {
            return Poly3::zero();
        }
        let mut out: BTreeMap<Exp, Rational> = BTreeMap::new();
        for (ea, ca) in &self.terms {
            for (eb, cb) in &other.terms {
                let e = exp_add(*ea, *eb);
                let entry = out.entry(e).or_insert_with(|| Rational::from(0));
                *entry += Rational::from(ca * cb);
            }
        }
        out.retain(|_, c| *c != 0);
        Poly3 { terms: out }
    }

    pub fn scale(&self, q: &Rational) -> Poly3 {
        if *q == 0 {
            return Poly3::zero();
        }
        Poly3 {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (*e, Rational::from(c * q)))
                .collect(),
        }
    }

    pub fn pow_u32(&self, e: u32) -> Poly3 {
        let mut acc = Poly3::one();
        let mut base = self.clone();
        let mut e = e;
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(&base);
            }
            base = base.mul(&base);
            e >>= 1;
        }
        acc
    }

    /// `p(axis ↦ axis + delta)`.
    pub fn shift(&self, axis: Axis, delta: i64) -> Poly3 {
        if delta == 0 || self.is_zero() {
            return self.clone();
        }
        let repl = Poly3::var(axis).add(&Poly3::from_i64(delta));
        self.eliminate_axis(axis, &repl)
    }

    /// `p(axis ↦ replacement)`, where `replacement` is any polynomial (it may
    /// reference `axis` itself, as [`Poly3::shift`] does, or eliminate it
    /// entirely, as the boundary substitution in
    /// [`super::boundary`] does).
    pub fn eliminate_axis(&self, axis: Axis, replacement: &Poly3) -> Poly3 {
        if self.is_zero() {
            return Poly3::zero();
        }
        // Memoize `replacement^e` across the distinct exponents actually used.
        let mut pow_cache: BTreeMap<u32, Poly3> = BTreeMap::new();
        pow_cache.insert(0, Poly3::one());
        let mut acc = Poly3::zero();
        for (e, c) in &self.terms {
            let ax_e = exp_axis(*e, axis);
            let rest = exp_with_axis(*e, axis, 0);
            let repl_pow = pow_cache
                .entry(ax_e)
                .or_insert_with(|| replacement.pow_u32(ax_e))
                .clone();
            let rest_mono = Poly3::monomial(rest, c.clone());
            acc = acc.add(&rest_mono.mul(&repl_pow));
        }
        acc
    }

    fn monomial(e: Exp, c: Rational) -> Poly3 {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(e, c);
        }
        Poly3 { terms }
    }

    /// Substitute a rational value for every axis at once. Used by this
    /// module's tests to check `shift`/`eliminate_axis` against direct
    /// evaluation, independent of how those are implemented.
    #[allow(dead_code)]
    pub fn eval(&self, n: &Rational, j: &Rational, k: &Rational) -> Rational {
        let mut acc = Rational::from(0);
        for ((en, ej, ek), c) in &self.terms {
            let mut term = c.clone();
            term *= n.clone().pow(*en);
            term *= j.clone().pow(*ej);
            term *= k.clone().pow(*ek);
            acc += term;
        }
        acc
    }

    pub fn to_expr(
        &self,
        pool: &crate::kernel::ExprPool,
        n: crate::kernel::ExprId,
        j: crate::kernel::ExprId,
        k: crate::kernel::ExprId,
    ) -> crate::kernel::ExprId {
        if self.terms.is_empty() {
            return pool.integer(0_i32);
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for ((en, ej, ek), c) in &self.terms {
            let mut factors = Vec::new();
            let ce = rational_to_expr(pool, c);
            let is_unit = *c == 1;
            if !is_unit || (*en == 0 && *ej == 0 && *ek == 0) {
                factors.push(ce);
            }
            push_pow(pool, &mut factors, n, *en);
            push_pow(pool, &mut factors, j, *ej);
            push_pow(pool, &mut factors, k, *ek);
            terms.push(if factors.len() == 1 {
                factors[0]
            } else {
                pool.mul(factors)
            });
        }
        if terms.len() == 1 {
            terms[0]
        } else {
            pool.add(terms)
        }
    }
}

fn push_pow(
    pool: &crate::kernel::ExprPool,
    factors: &mut Vec<crate::kernel::ExprId>,
    var: crate::kernel::ExprId,
    e: u32,
) {
    match e {
        0 => {}
        1 => factors.push(var),
        _ => factors.push(pool.pow(var, pool.integer(e as i64))),
    }
}

fn rational_to_expr(pool: &crate::kernel::ExprPool, q: &Rational) -> crate::kernel::ExprId {
    let (num, den) = (q.numer().clone(), q.denom().clone());
    if den == 1 {
        pool.integer(num)
    } else {
        pool.rational(num, den)
    }
}

/// An element of `Q(n, j, k)`, kept as a raw (not necessarily reduced)
/// `num/den` pair. See the module docs for why no gcd reduction happens here.
#[derive(Clone, Debug)]
pub struct Rat3 {
    pub num: Poly3,
    pub den: Poly3,
}

impl Rat3 {
    pub fn zero() -> Self {
        Rat3 {
            num: Poly3::zero(),
            den: Poly3::one(),
        }
    }

    pub fn one() -> Self {
        Rat3 {
            num: Poly3::one(),
            den: Poly3::one(),
        }
    }

    pub fn from_poly(p: Poly3) -> Self {
        Rat3 {
            num: p,
            den: Poly3::one(),
        }
    }

    pub fn from_rational(q: Rational) -> Self {
        Rat3::from_poly(Poly3::constant(q))
    }

    pub fn add(&self, other: &Rat3) -> Rat3 {
        Rat3 {
            num: self.num.mul(&other.den).add(&other.num.mul(&self.den)),
            den: self.den.mul(&other.den),
        }
    }

    pub fn neg(&self) -> Rat3 {
        Rat3 {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    pub fn sub(&self, other: &Rat3) -> Rat3 {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Rat3) -> Rat3 {
        Rat3 {
            num: self.num.mul(&other.num),
            den: self.den.mul(&other.den),
        }
    }

    pub fn inv(&self) -> Option<Rat3> {
        if self.num.is_zero() {
            return None;
        }
        Some(Rat3 {
            num: self.den.clone(),
            den: self.num.clone(),
        })
    }

    pub fn div(&self, other: &Rat3) -> Option<Rat3> {
        Some(self.mul(&other.inv()?))
    }

    pub fn pow_i32(&self, e: i32) -> Option<Rat3> {
        if e == 0 {
            return Some(Rat3::one());
        }
        let base = if e < 0 { self.inv()? } else { self.clone() };
        Some(Rat3 {
            num: base.num.pow_u32(e.unsigned_abs()),
            den: base.den.pow_u32(e.unsigned_abs()),
        })
    }

    pub fn shift(&self, axis: Axis, delta: i64) -> Rat3 {
        Rat3 {
            num: self.num.shift(axis, delta),
            den: self.den.shift(axis, delta),
        }
    }

    pub fn to_expr(
        &self,
        pool: &crate::kernel::ExprPool,
        n: crate::kernel::ExprId,
        j: crate::kernel::ExprId,
        k: crate::kernel::ExprId,
    ) -> crate::kernel::ExprId {
        let num = self.num.to_expr(pool, n, j, k);
        if self.den.is_one() {
            return num;
        }
        let den = self.den.to_expr(pool, n, j, k);
        pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
    }

    /// `self == other` as an identity of rational functions: cross-multiply
    /// and compare numerators. Exact, no reduction needed.
    pub fn eq_rat(&self, other: &Rat3) -> bool {
        self.num
            .mul(&other.den)
            .sub(&other.num.mul(&self.den))
            .is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly3_basic_arithmetic() {
        let n = Poly3::var(Axis::N);
        let j = Poly3::var(Axis::J);
        let one = Poly3::one();
        let p = n.add(&j).add(&one); // n + j + 1
        let q = n.sub(&j); // n - j
        let prod = p.mul(&q); // n^2 - j^2 + n - j
        assert_eq!(prod.degree(Axis::N), 2);
        assert_eq!(prod.degree(Axis::J), 2);
        // n^2 - j^2 + n - j at (n,j,k)=(3,2,0): 9 - 4 + 3 - 2 = 6.
        assert_eq!(
            prod.eval(&Rational::from(3), &Rational::from(2), &Rational::from(0)),
            Rational::from(6)
        );
    }

    #[test]
    fn shift_matches_direct_eval() {
        // p = n*k + j; p(k -> k+2) evaluated at (n,j,k)=(2,3,5) should equal
        // p evaluated at (2,3,7).
        let n = Poly3::var(Axis::N);
        let j = Poly3::var(Axis::J);
        let k = Poly3::var(Axis::K);
        let p = n.mul(&k).add(&j);
        let shifted = p.shift(Axis::K, 2);
        let a = shifted.eval(&Rational::from(2), &Rational::from(3), &Rational::from(5));
        let b = p.eval(&Rational::from(2), &Rational::from(3), &Rational::from(7));
        assert_eq!(a, b);
    }

    #[test]
    fn eliminate_axis_matches_composition() {
        // p = j^2 + k, replace j by (2n+1). At n=3 that's j=7.
        let j = Poly3::var(Axis::J);
        let k = Poly3::var(Axis::K);
        let n = Poly3::var(Axis::N);
        let p = j.mul(&j).add(&k);
        let repl = n.scale(&Rational::from(2)).add(&Poly3::one());
        let sub = p.eliminate_axis(Axis::J, &repl);
        let a = sub.eval(&Rational::from(3), &Rational::from(0), &Rational::from(5));
        let b = p.eval(&Rational::from(3), &Rational::from(7), &Rational::from(5));
        assert_eq!(a, b);
    }

    #[test]
    fn rat3_cross_mul_zero_test() {
        let n = Poly3::var(Axis::N);
        let one = Poly3::one();
        let a = Rat3::from_poly(n.clone())
            .div(&Rat3::from_poly(n.add(&one)))
            .unwrap();
        let b = Rat3 {
            num: n.mul(&Poly3::from_i64(2)),
            den: n.add(&one).mul(&Poly3::from_i64(2)),
        };
        assert!(a.eq_rat(&b));
    }
}
