//! Exact sparse arithmetic in `Q[n, x_1, …, x_m]` and `Q(n, x_1, …, x_m)` for
//! an arbitrary number `m ≥ 1` of bound indices.
//!
//! The single-sum engine ([`super::super::qfield`]) builds a careful tower
//! `Q(n)[k]` / `Q(n)(k)` with a subresultant-PRS gcd, because Gosper's
//! algorithm needs a genuine normal form. The ansatz search in
//! [`super::search`] never calls a gcd: it fixes the certificate
//! denominators up front from the term's own shift-ratio denominators (see
//! that module's docs) and solves a *linear* system for the numerators. So
//! this module is deliberately simpler than `qfield` — a plain sparse
//! multivariate polynomial ring over `Q`, dense enough operations
//! (add/mul/shift/substitute) and nothing else. [`RatM`] (the field of
//! fractions) is kept unreduced throughout; the only place that matters is
//! the final exact identity check, which is a zero-test on a cross-multiplied
//! numerator, not a canonical form.
//!
//! # Axis convention
//!
//! Every polynomial here lives in `m + 1` variables: `n` is always axis `0`
//! ([`AXIS_N`]), and the `m` bound indices are axes `1..=m`, in the order the
//! caller supplied them. `num_axes = m + 1` must be threaded consistently by
//! every caller that constructs a fresh polynomial ([`PolyM::var`],
//! [`PolyM::constant`], [`PolyM::one`]) — this module does not itself track
//! `m` (a `PolyM` is just a sparse exponent map), so a caller that mixes
//! polynomials built with different `num_axes` gets a silently wrong
//! (mismatched-length) exponent vector; every caller in this crate threads a
//! single `num_axes` value throughout one search.

use rug::ops::Pow as _;
use rug::Rational;
use std::collections::BTreeMap;

/// Which variable an operation acts on: `0` is always `n`; `1..=m` are the
/// bound indices, in caller-supplied order.
pub type Axis = usize;

/// `n` is always axis `0`.
pub const AXIS_N: Axis = 0;

/// Exponents `(deg_n, deg_x1, …, deg_xm)` of one monomial — length `m + 1`.
pub type Exp = Vec<u32>;

fn exp_with_axis(mut e: Exp, axis: Axis, v: u32) -> Exp {
    e[axis] = v;
    e
}

fn exp_add(a: &Exp, b: &Exp) -> Exp {
    debug_assert_eq!(a.len(), b.len(), "PolyM operands must share num_axes");
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// A polynomial in `Q[n, x_1, …, x_m]`, stored as a sparse map from exponent
/// tuples to non-zero rational coefficients.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PolyM {
    pub terms: BTreeMap<Exp, Rational>,
}

impl PolyM {
    pub fn zero() -> Self {
        PolyM {
            terms: BTreeMap::new(),
        }
    }

    pub fn one(num_axes: usize) -> Self {
        PolyM::constant(Rational::from(1), num_axes)
    }

    pub fn constant(q: Rational, num_axes: usize) -> Self {
        let mut terms = BTreeMap::new();
        if q != 0 {
            terms.insert(vec![0u32; num_axes], q);
        }
        PolyM { terms }
    }

    pub fn from_i64(v: i64, num_axes: usize) -> Self {
        PolyM::constant(Rational::from(v), num_axes)
    }

    pub fn var(axis: Axis, num_axes: usize) -> Self {
        let e = exp_with_axis(vec![0u32; num_axes], axis, 1);
        let mut terms = BTreeMap::new();
        terms.insert(e, Rational::from(1));
        PolyM { terms }
    }

    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn is_one(&self) -> bool {
        self.terms.len() == 1
            && self
                .terms
                .iter()
                .next()
                .map(|(e, c)| e.iter().all(|&x| x == 0) && *c == 1)
                .unwrap_or(false)
    }

    /// The as-constructed constant value, or `None` if `self` is not
    /// (syntactically) a constant.
    pub fn as_constant(&self) -> Option<Rational> {
        if self.terms.is_empty() {
            return Some(Rational::from(0));
        }
        if self.terms.len() == 1 {
            if let Some((e, c)) = self.terms.iter().next() {
                if e.iter().all(|&x| x == 0) {
                    return Some(c.clone());
                }
            }
        }
        None
    }

    /// Highest exponent on `axis` among the non-zero terms, or `-1` for the
    /// zero polynomial.
    #[allow(dead_code)]
    pub fn degree(&self, axis: Axis) -> i64 {
        self.terms
            .keys()
            .map(|e| e[axis] as i64)
            .max()
            .unwrap_or(-1)
    }

    pub fn neg(&self) -> PolyM {
        PolyM {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), Rational::from(-c)))
                .collect(),
        }
    }

    pub fn add(&self, other: &PolyM) -> PolyM {
        let mut out = self.terms.clone();
        for (e, c) in &other.terms {
            let entry = out.entry(e.clone()).or_insert_with(|| Rational::from(0));
            *entry += c;
        }
        out.retain(|_, c| *c != 0);
        PolyM { terms: out }
    }

    pub fn sub(&self, other: &PolyM) -> PolyM {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &PolyM) -> PolyM {
        if self.is_zero() || other.is_zero() {
            return PolyM::zero();
        }
        let mut out: BTreeMap<Exp, Rational> = BTreeMap::new();
        for (ea, ca) in &self.terms {
            for (eb, cb) in &other.terms {
                let e = exp_add(ea, eb);
                let entry = out.entry(e).or_insert_with(|| Rational::from(0));
                *entry += Rational::from(ca * cb);
            }
        }
        out.retain(|_, c| *c != 0);
        PolyM { terms: out }
    }

    pub fn scale(&self, q: &Rational) -> PolyM {
        if *q == 0 {
            return PolyM::zero();
        }
        PolyM {
            terms: self
                .terms
                .iter()
                .map(|(e, c)| (e.clone(), Rational::from(c * q)))
                .collect(),
        }
    }

    pub fn pow_u32(&self, e: u32) -> PolyM {
        let num_axes = self.terms.keys().next().map(|k| k.len());
        let mut acc = match num_axes {
            Some(na) => PolyM::one(na),
            // self is the zero polynomial; 0^0 = 1 has no well-defined
            // `num_axes` here, but `pow_u32` is never called with e == 0 on
            // a zero polynomial in this crate's actual call sites, and the
            // caller-visible behavior (zero raised to any positive power is
            // zero) is preserved by the loop below regardless.
            None => PolyM::zero(),
        };
        if e == 0 && num_axes.is_none() {
            return PolyM::zero();
        }
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
    pub fn shift(&self, axis: Axis, delta: i64) -> PolyM {
        if delta == 0 || self.is_zero() {
            return self.clone();
        }
        let num_axes = self.terms.keys().next().unwrap().len();
        let repl = PolyM::var(axis, num_axes).add(&PolyM::from_i64(delta, num_axes));
        self.eliminate_axis(axis, &repl)
    }

    /// `p(axis ↦ replacement)`, where `replacement` is any polynomial (it may
    /// reference `axis` itself, as [`PolyM::shift`] does, or eliminate it
    /// entirely, as the boundary substitution in [`super::boundary`] does).
    pub fn eliminate_axis(&self, axis: Axis, replacement: &PolyM) -> PolyM {
        if self.is_zero() {
            return PolyM::zero();
        }
        // Memoize `replacement^e` across the distinct exponents actually used.
        let mut pow_cache: BTreeMap<u32, PolyM> = BTreeMap::new();
        let num_axes = self.terms.keys().next().unwrap().len();
        pow_cache.insert(0, PolyM::one(num_axes));
        let mut acc = PolyM::zero();
        for (e, c) in &self.terms {
            let ax_e = e[axis];
            let rest = exp_with_axis(e.clone(), axis, 0);
            let repl_pow = pow_cache
                .entry(ax_e)
                .or_insert_with(|| replacement.pow_u32(ax_e))
                .clone();
            let rest_mono = PolyM::monomial(rest, c.clone());
            acc = acc.add(&rest_mono.mul(&repl_pow));
        }
        acc
    }

    fn monomial(e: Exp, c: Rational) -> PolyM {
        let mut terms = BTreeMap::new();
        if c != 0 {
            terms.insert(e, c);
        }
        PolyM { terms }
    }

    /// Substitute a rational value for every axis at once (`vals[0]` for
    /// `n`, `vals[1..]` for the bound indices in order). Used by this
    /// module's tests to check `shift`/`eliminate_axis` against direct
    /// evaluation, independent of how those are implemented.
    #[allow(dead_code)]
    pub fn eval(&self, vals: &[Rational]) -> Rational {
        let mut acc = Rational::from(0);
        for (e, c) in &self.terms {
            let mut term = c.clone();
            for (ax, &exp) in e.iter().enumerate() {
                term *= vals[ax].clone().pow(exp);
            }
            acc += term;
        }
        acc
    }

    pub fn to_expr(
        &self,
        pool: &crate::kernel::ExprPool,
        n: crate::kernel::ExprId,
        indices: &[crate::kernel::ExprId],
    ) -> crate::kernel::ExprId {
        if self.terms.is_empty() {
            return pool.integer(0_i32);
        }
        let mut terms = Vec::with_capacity(self.terms.len());
        for (e, c) in &self.terms {
            let mut factors = Vec::new();
            let ce = rational_to_expr(pool, c);
            let is_unit = *c == 1;
            let all_zero = e.iter().all(|&x| x == 0);
            if !is_unit || all_zero {
                factors.push(ce);
            }
            push_pow(pool, &mut factors, n, e[0]);
            for (t, &idx_expr) in indices.iter().enumerate() {
                push_pow(pool, &mut factors, idx_expr, e[t + 1]);
            }
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

/// An element of `Q(n, x_1, …, x_m)`, kept as a raw (not necessarily reduced)
/// `num/den` pair. See the module docs for why no gcd reduction happens here.
#[derive(Clone, Debug)]
pub struct RatM {
    pub num: PolyM,
    pub den: PolyM,
}

impl RatM {
    pub fn one(num_axes: usize) -> Self {
        RatM {
            num: PolyM::one(num_axes),
            den: PolyM::one(num_axes),
        }
    }

    /// `num_axes` **must** be supplied explicitly rather than inferred from
    /// `p`'s own keys: `p` may be the zero polynomial (empty term map, no
    /// exponent-vector length to read off), and a `den` built with the wrong
    /// `num_axes` silently mismatches every other polynomial's exponent
    /// vectors the first time it's multiplied against one — exactly the bug
    /// this explicit parameter exists to rule out at the type level.
    pub fn from_poly(p: PolyM, num_axes: usize) -> Self {
        RatM {
            num: p,
            den: PolyM::one(num_axes),
        }
    }

    pub fn from_rational(q: Rational, num_axes: usize) -> Self {
        RatM::from_poly(PolyM::constant(q, num_axes), num_axes)
    }

    pub fn add(&self, other: &RatM) -> RatM {
        RatM {
            num: self.num.mul(&other.den).add(&other.num.mul(&self.den)),
            den: self.den.mul(&other.den),
        }
    }

    pub fn neg(&self) -> RatM {
        RatM {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    pub fn sub(&self, other: &RatM) -> RatM {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &RatM) -> RatM {
        RatM {
            num: self.num.mul(&other.num),
            den: self.den.mul(&other.den),
        }
    }

    pub fn inv(&self) -> Option<RatM> {
        if self.num.is_zero() {
            return None;
        }
        Some(RatM {
            num: self.den.clone(),
            den: self.num.clone(),
        })
    }

    pub fn div(&self, other: &RatM) -> Option<RatM> {
        Some(self.mul(&other.inv()?))
    }

    pub fn pow_i32(&self, e: i32) -> Option<RatM> {
        if e == 0 {
            let num_axes = self
                .num
                .terms
                .keys()
                .next()
                .or_else(|| self.den.terms.keys().next())
                .map(|k| k.len())
                .unwrap_or(1);
            return Some(RatM::one(num_axes));
        }
        let base = if e < 0 { self.inv()? } else { self.clone() };
        Some(RatM {
            num: base.num.pow_u32(e.unsigned_abs()),
            den: base.den.pow_u32(e.unsigned_abs()),
        })
    }

    pub fn shift(&self, axis: Axis, delta: i64) -> RatM {
        RatM {
            num: self.num.shift(axis, delta),
            den: self.den.shift(axis, delta),
        }
    }

    pub fn to_expr(
        &self,
        pool: &crate::kernel::ExprPool,
        n: crate::kernel::ExprId,
        indices: &[crate::kernel::ExprId],
    ) -> crate::kernel::ExprId {
        let num = self.num.to_expr(pool, n, indices);
        if self.den.is_one() {
            return num;
        }
        let den = self.den.to_expr(pool, n, indices);
        pool.mul(vec![num, pool.pow(den, pool.integer(-1_i32))])
    }

    /// `self == other` as an identity of rational functions: cross-multiply
    /// and compare numerators. Exact, no reduction needed.
    pub fn eq_rat(&self, other: &RatM) -> bool {
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
    fn polym_basic_arithmetic() {
        // 3 axes: n, j, k (num_axes = 3).
        let n = PolyM::var(0, 3);
        let j = PolyM::var(1, 3);
        let one = PolyM::one(3);
        let p = n.add(&j).add(&one); // n + j + 1
        let q = n.sub(&j); // n - j
        let prod = p.mul(&q); // n^2 - j^2 + n - j
        assert_eq!(prod.degree(0), 2);
        assert_eq!(prod.degree(1), 2);
        // n^2 - j^2 + n - j at (n,j,k)=(3,2,0): 9 - 4 + 3 - 2 = 6.
        let vals = [Rational::from(3), Rational::from(2), Rational::from(0)];
        assert_eq!(prod.eval(&vals), Rational::from(6));
    }

    #[test]
    fn shift_matches_direct_eval() {
        // p = n*k + j; p(k -> k+2) evaluated at (n,j,k)=(2,3,5) should equal
        // p evaluated at (2,3,7). Axes: n=0, j=1, k=2.
        let n = PolyM::var(0, 3);
        let j = PolyM::var(1, 3);
        let k = PolyM::var(2, 3);
        let p = n.mul(&k).add(&j);
        let shifted = p.shift(2, 2);
        let a = shifted.eval(&[Rational::from(2), Rational::from(3), Rational::from(5)]);
        let b = p.eval(&[Rational::from(2), Rational::from(3), Rational::from(7)]);
        assert_eq!(a, b);
    }

    #[test]
    fn eliminate_axis_matches_composition() {
        // p = j^2 + k, replace j by (2n+1). At n=3 that's j=7.
        let j = PolyM::var(1, 3);
        let k = PolyM::var(2, 3);
        let n = PolyM::var(0, 3);
        let p = j.mul(&j).add(&k);
        let repl = n.scale(&Rational::from(2)).add(&PolyM::one(3));
        let sub = p.eliminate_axis(1, &repl);
        let a = sub.eval(&[Rational::from(3), Rational::from(0), Rational::from(5)]);
        let b = p.eval(&[Rational::from(3), Rational::from(7), Rational::from(5)]);
        assert_eq!(a, b);
    }

    #[test]
    fn ratm_cross_mul_zero_test() {
        let n = PolyM::var(0, 3);
        let one = PolyM::one(3);
        let a = RatM::from_poly(n.clone(), 3)
            .div(&RatM::from_poly(n.add(&one), 3))
            .unwrap();
        let b = RatM {
            num: n.mul(&PolyM::from_i64(2, 3)),
            den: n.add(&one).mul(&PolyM::from_i64(2, 3)),
        };
        assert!(a.eq_rat(&b));
    }

    /// Exercise a fourth axis (an `m = 3` bound-index arity, `num_axes = 4`)
    /// to make sure nothing in this module secretly assumes exactly three
    /// axes.
    #[test]
    fn four_axis_arithmetic_works() {
        let n = PolyM::var(0, 4);
        let x1 = PolyM::var(1, 4);
        let x2 = PolyM::var(2, 4);
        let x3 = PolyM::var(3, 4);
        let p = n.add(&x1).add(&x2).add(&x3); // n + x1 + x2 + x3
        let vals = [
            Rational::from(1),
            Rational::from(2),
            Rational::from(3),
            Rational::from(4),
        ];
        assert_eq!(p.eval(&vals), Rational::from(10));
        let shifted = p.shift(3, 5); // x3 -> x3 + 5
        assert_eq!(shifted.eval(&vals), Rational::from(15));
    }
}
