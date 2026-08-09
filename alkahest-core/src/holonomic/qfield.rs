//! Exact arithmetic in `Q(n)` and `Q(n)(k)`.
//!
//! Creative telescoping needs a *field* of coefficients for the linear algebra
//! (the unknowns `a_i(n)` and the certificate coefficients live in `Q(n)`) and a
//! ring of polynomials over it (the `k`-side of the Gosper equation).  This
//! module supplies both, on top of [`RatUniPoly`] (dense `Q[x]`) and
//! [`RatFunc`] (reduced `Q(x)`), which already exist for Gosper summation.
//!
//! Nothing here is approximate: every operation is exact rational arithmetic,
//! which is what makes the final certificate check in
//! [`super::zeilberger::zeilberger()`] a proof rather than a spot check.

use crate::matrix::normal_form::RatUniPoly;
use crate::sum::RatFunc;
use rug::{Integer, Rational};

/// An element of `Q(n)` — a reduced rational function in the outer variable.
pub type Rn = RatFunc;

pub fn rn_zero() -> Rn {
    RatFunc::zero()
}

pub fn rn_one() -> Rn {
    RatFunc::one()
}

pub fn rn_int(i: i64) -> Rn {
    RatFunc::scalar(Rational::from(i))
}

pub fn rn_rat(q: Rational) -> Rn {
    RatFunc::scalar(q)
}

/// The generator `n` itself.
pub fn rn_var() -> Rn {
    RatFunc::from_poly(RatUniPoly::x())
}

pub fn rn_poly(p: RatUniPoly) -> Rn {
    RatFunc::from_poly(p).normalize()
}

pub fn rn_is_zero(a: &Rn) -> bool {
    a.num.is_zero()
}

pub fn rn_add(a: &Rn, b: &Rn) -> Rn {
    a.clone() + b.clone()
}

pub fn rn_neg(a: &Rn) -> Rn {
    -a.clone()
}

pub fn rn_sub(a: &Rn, b: &Rn) -> Rn {
    a.clone() + (-b.clone())
}

pub fn rn_mul(a: &Rn, b: &Rn) -> Rn {
    a.mul_ratfunc(b)
}

pub fn rn_inv(a: &Rn) -> Option<Rn> {
    a.inv()
}

pub fn rn_div(a: &Rn, b: &Rn) -> Option<Rn> {
    Some(rn_mul(a, &rn_inv(b)?))
}

pub fn rn_eq(a: &Rn, b: &Rn) -> bool {
    rn_is_zero(&rn_sub(a, b))
}

/// `a(n + i)`.
pub fn rn_shift(a: &Rn, i: i64) -> Rn {
    if i == 0 {
        return a.clone();
    }
    a.compose_affine_arg(&Rational::from(1), &Rational::from(i))
}

fn poly_deriv(p: &RatUniPoly) -> RatUniPoly {
    if p.coeffs.len() <= 1 {
        return RatUniPoly::zero();
    }
    let coeffs: Vec<Rational> = p
        .coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c.clone() * Rational::from(i as i64))
        .collect();
    RatUniPoly { coeffs }.trim()
}

/// `d/dn` of a rational function.
pub fn rn_deriv(a: &Rn) -> Rn {
    let nu = poly_deriv(&a.num);
    let dv = poly_deriv(&a.den);
    let num = &(&nu * &a.den) - &(&a.num * &dv);
    let den = &a.den * &a.den;
    RatFunc { num, den }.normalize()
}

fn poly_eval(p: &RatUniPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.coeffs.iter().rev() {
        acc *= x.clone();
        acc += c.clone();
    }
    acc
}

/// Evaluate at a rational point; `None` at a pole.
pub fn rn_eval(a: &Rn, x: &Rational) -> Option<Rational> {
    let d = poly_eval(&a.den, x);
    if d == 0 {
        return None;
    }
    Some(poly_eval(&a.num, x) / d)
}

/// Clear denominators across a slice of `Q(n)` elements, returning integer
/// primitive polynomials in `n` that are proportional to the input.
///
/// The scaling is common to all entries, so a linear relation with these
/// coefficients holds exactly when it held for the input.
pub fn clear_denominators(items: &[Rn]) -> Vec<RatUniPoly> {
    // Common denominator: product of denominators reduced by pairwise gcd.
    let mut common = RatUniPoly::one();
    for it in items {
        if it.den.is_zero() {
            continue;
        }
        let g = common.gcd(&it.den);
        let (q, _) = RatUniPoly::div_rem(&it.den, &g);
        common = &common * &q;
    }
    let mut out: Vec<RatUniPoly> = Vec::with_capacity(items.len());
    for it in items {
        let (mult, _) = RatUniPoly::div_rem(&common, &it.den);
        out.push((&it.num * &mult).trim());
    }
    make_primitive(&mut out);
    out
}

/// Scale a family of rational polynomials by one common rational so that all
/// coefficients are integers with overall content 1 and a positive leading
/// coefficient on the last non-zero entry.
pub fn make_primitive(polys: &mut [RatUniPoly]) {
    let mut den_lcm = Integer::from(1);
    for p in polys.iter() {
        for c in &p.coeffs {
            den_lcm = den_lcm.lcm(&c.clone().denom().clone());
        }
    }
    let scale = Rational::from(den_lcm);
    for p in polys.iter_mut() {
        for c in p.coeffs.iter_mut() {
            *c *= scale.clone();
        }
    }
    let mut content = Integer::from(0);
    for p in polys.iter() {
        for c in &p.coeffs {
            content = content.gcd(&c.clone().numer().clone());
        }
    }
    if content != 0 && content != 1 {
        let inv = Rational::from((Integer::from(1), content));
        for p in polys.iter_mut() {
            for c in p.coeffs.iter_mut() {
                *c *= inv.clone();
            }
        }
    }
    let sign_ref = polys
        .iter()
        .rev()
        .find(|p| !p.is_zero())
        .map(|p| p.leading_coeff());
    if let Some(lc) = sign_ref {
        if lc < 0 {
            for p in polys.iter_mut() {
                for c in p.coeffs.iter_mut() {
                    *c *= Rational::from(-1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Q(n)[k]
// ---------------------------------------------------------------------------

/// A polynomial in `k` with coefficients in `Q(n)` (ascending order).
#[derive(Clone, Debug)]
pub struct PolyK {
    pub coeffs: Vec<Rn>,
}

impl PolyK {
    pub fn zero() -> Self {
        PolyK { coeffs: vec![] }
    }

    pub fn one() -> Self {
        PolyK {
            coeffs: vec![rn_one()],
        }
    }

    pub fn constant(c: Rn) -> Self {
        PolyK { coeffs: vec![c] }.trim()
    }

    /// The polynomial `k`.
    pub fn k() -> Self {
        PolyK {
            coeffs: vec![rn_zero(), rn_one()],
        }
    }

    pub fn from_coeffs(coeffs: Vec<Rn>) -> Self {
        PolyK { coeffs }.trim()
    }

    pub fn trim(mut self) -> Self {
        while self.coeffs.last().map(rn_is_zero).unwrap_or(false) {
            self.coeffs.pop();
        }
        self
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(rn_is_zero)
    }

    /// Degree, or `-1` for the zero polynomial.
    pub fn degree(&self) -> i32 {
        let mut d = self.coeffs.len() as i32 - 1;
        while d >= 0 && rn_is_zero(&self.coeffs[d as usize]) {
            d -= 1;
        }
        d
    }

    pub fn coeff(&self, i: usize) -> Rn {
        self.coeffs.get(i).cloned().unwrap_or_else(rn_zero)
    }

    pub fn leading_coeff(&self) -> Rn {
        let d = self.degree();
        if d < 0 {
            rn_zero()
        } else {
            self.coeff(d as usize)
        }
    }

    pub fn add(&self, other: &PolyK) -> PolyK {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(rn_add(&self.coeff(i), &other.coeff(i)));
        }
        PolyK { coeffs: out }.trim()
    }

    pub fn neg(&self) -> PolyK {
        PolyK {
            coeffs: self.coeffs.iter().map(rn_neg).collect(),
        }
    }

    pub fn sub(&self, other: &PolyK) -> PolyK {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &PolyK) -> PolyK {
        if self.is_zero() || other.is_zero() {
            return PolyK::zero();
        }
        let mut out = vec![rn_zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if rn_is_zero(a) {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                if rn_is_zero(b) {
                    continue;
                }
                out[i + j] = rn_add(&out[i + j], &rn_mul(a, b));
            }
        }
        PolyK { coeffs: out }.trim()
    }

    pub fn scale(&self, c: &Rn) -> PolyK {
        if rn_is_zero(c) {
            return PolyK::zero();
        }
        PolyK {
            coeffs: self.coeffs.iter().map(|a| rn_mul(a, c)).collect(),
        }
        .trim()
    }

    /// Euclidean division over the field `Q(n)`.
    pub fn div_rem(a: &PolyK, b: &PolyK) -> Option<(PolyK, PolyK)> {
        if b.is_zero() {
            return None;
        }
        let db = b.degree();
        let lb = b.leading_coeff();
        let lb_inv = rn_inv(&lb)?;
        let mut rem = a.clone().trim();
        let mut quot = vec![rn_zero(); ((a.degree() - db).max(-1) + 1).max(0) as usize];
        while rem.degree() >= db && !rem.is_zero() {
            let shift = (rem.degree() - db) as usize;
            let t = rn_mul(&rem.leading_coeff(), &lb_inv);
            if shift >= quot.len() {
                quot.resize(shift + 1, rn_zero());
            }
            quot[shift] = rn_add(&quot[shift], &t);
            let mut sub_coeffs = vec![rn_zero(); shift];
            sub_coeffs.extend(b.coeffs.iter().map(|c| rn_mul(c, &t)));
            let sub = PolyK { coeffs: sub_coeffs };
            rem = rem.sub(&sub);
        }
        Some((PolyK { coeffs: quot }.trim(), rem.trim()))
    }

    pub fn exact_div(a: &PolyK, b: &PolyK) -> Option<PolyK> {
        let (q, r) = PolyK::div_rem(a, b)?;
        if r.is_zero() {
            Some(q)
        } else {
            None
        }
    }

    /// Monic gcd over `Q(n)`.
    pub fn gcd(a: &PolyK, b: &PolyK) -> PolyK {
        let mut x = a.clone().trim();
        let mut y = b.clone().trim();
        if x.degree() < y.degree() {
            std::mem::swap(&mut x, &mut y);
        }
        while !y.is_zero() {
            let Some((_, r)) = PolyK::div_rem(&x, &y) else {
                return PolyK::one();
            };
            x = y;
            y = r;
        }
        if x.is_zero() {
            PolyK::zero()
        } else {
            x.monic()
        }
    }

    pub fn monic(&self) -> PolyK {
        let lc = self.leading_coeff();
        match rn_inv(&lc) {
            Some(inv) => self.scale(&inv),
            None => self.clone(),
        }
    }

    /// `p(k + j)`.
    pub fn shift_k(&self, j: i64) -> PolyK {
        if j == 0 || self.is_zero() {
            return self.clone().trim();
        }
        let kj = PolyK {
            coeffs: vec![rn_int(j), rn_one()],
        };
        let mut acc = PolyK::zero();
        let mut pow = PolyK::one();
        for c in &self.coeffs {
            acc = acc.add(&pow.scale(c));
            pow = pow.mul(&kj);
        }
        acc.trim()
    }

    /// `p` with `n ↦ n + i` applied to every coefficient.
    pub fn shift_n(&self, i: i64) -> PolyK {
        if i == 0 {
            return self.clone();
        }
        PolyK {
            coeffs: self.coeffs.iter().map(|c| rn_shift(c, i)).collect(),
        }
        .trim()
    }

    pub fn eq_poly(&self, other: &PolyK) -> bool {
        self.sub(other).is_zero()
    }

    /// `lcm` via `a·b/gcd`.
    pub fn lcm(a: &PolyK, b: &PolyK) -> PolyK {
        if a.is_zero() || b.is_zero() {
            return PolyK::zero();
        }
        let g = PolyK::gcd(a, b);
        let prod = a.mul(b);
        PolyK::exact_div(&prod, &g).unwrap_or(prod)
    }
}

// ---------------------------------------------------------------------------
// Q(n)(k)
// ---------------------------------------------------------------------------

/// A rational function in `k` over `Q(n)` — i.e. an element of `Q(n, k)`.
#[derive(Clone, Debug)]
pub struct RatK {
    pub num: PolyK,
    pub den: PolyK,
}

impl RatK {
    pub fn zero() -> Self {
        RatK {
            num: PolyK::zero(),
            den: PolyK::one(),
        }
    }

    pub fn one() -> Self {
        RatK {
            num: PolyK::one(),
            den: PolyK::one(),
        }
    }

    pub fn from_poly(p: PolyK) -> Self {
        RatK {
            num: p,
            den: PolyK::one(),
        }
        .normalize()
    }

    pub fn from_rn(c: Rn) -> Self {
        RatK::from_poly(PolyK::constant(c))
    }

    pub fn k() -> Self {
        RatK::from_poly(PolyK::k())
    }

    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    pub fn normalize(mut self) -> Self {
        if self.num.is_zero() {
            return RatK::zero();
        }
        if self.den.is_zero() {
            return self;
        }
        let g = PolyK::gcd(&self.num, &self.den);
        if g.degree() > 0 {
            if let Some(n2) = PolyK::exact_div(&self.num, &g) {
                if let Some(d2) = PolyK::exact_div(&self.den, &g) {
                    self.num = n2;
                    self.den = d2;
                }
            }
        }
        // Make the denominator monic in k with a `Q(n)` leading coefficient of 1.
        let lc = self.den.leading_coeff();
        if let Some(inv) = rn_inv(&lc) {
            self.num = self.num.scale(&inv);
            self.den = self.den.scale(&inv);
        }
        self
    }

    pub fn add(&self, other: &RatK) -> RatK {
        RatK {
            num: self.num.mul(&other.den).add(&other.num.mul(&self.den)),
            den: self.den.mul(&other.den),
        }
        .normalize()
    }

    pub fn neg(&self) -> RatK {
        RatK {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    pub fn sub(&self, other: &RatK) -> RatK {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &RatK) -> RatK {
        RatK {
            num: self.num.mul(&other.num),
            den: self.den.mul(&other.den),
        }
        .normalize()
    }

    pub fn inv(&self) -> Option<RatK> {
        if self.num.is_zero() {
            return None;
        }
        Some(
            RatK {
                num: self.den.clone(),
                den: self.num.clone(),
            }
            .normalize(),
        )
    }

    pub fn div(&self, other: &RatK) -> Option<RatK> {
        Some(self.mul(&other.inv()?))
    }

    pub fn pow_i32(&self, e: i32) -> Option<RatK> {
        if e == 0 {
            return Some(RatK::one());
        }
        let base = if e < 0 { self.inv()? } else { self.clone() };
        let mut acc = RatK::one();
        for _ in 0..e.unsigned_abs() {
            acc = acc.mul(&base);
        }
        Some(acc)
    }

    pub fn shift_k(&self, j: i64) -> RatK {
        RatK {
            num: self.num.shift_k(j),
            den: self.den.shift_k(j),
        }
        .normalize()
    }

    pub fn shift_n(&self, i: i64) -> RatK {
        RatK {
            num: self.num.shift_n(i),
            den: self.den.shift_n(i),
        }
        .normalize()
    }

    pub fn eq_ratk(&self, other: &RatK) -> bool {
        self.sub(other).is_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rn_of(coeffs: &[i64]) -> Rn {
        rn_poly(
            RatUniPoly {
                coeffs: coeffs.iter().map(|c| Rational::from(*c)).collect(),
            }
            .trim(),
        )
    }

    #[test]
    fn rn_basic_field_ops() {
        let n = rn_var();
        let one = rn_one();
        let a = rn_add(&n, &one); // n + 1
        let b = rn_sub(&n, &one); // n - 1
        let prod = rn_mul(&a, &b); // n^2 - 1
        assert!(rn_eq(&prod, &rn_of(&[-1, 0, 1])));
        let q = rn_div(&prod, &a).expect("nonzero divisor");
        assert!(rn_eq(&q, &b));
        assert!(rn_eq(&rn_shift(&n, 2), &rn_of(&[2, 1])));
    }

    #[test]
    fn rn_derivative_and_eval() {
        // d/dn (1/n) = -1/n^2
        let n = rn_var();
        let inv = rn_inv(&n).expect("n != 0");
        let d = rn_deriv(&inv);
        let expected = rn_neg(&rn_inv(&rn_mul(&n, &n)).unwrap());
        assert!(rn_eq(&d, &expected));
        assert_eq!(
            rn_eval(&inv, &Rational::from(4)).unwrap(),
            Rational::from((1, 4))
        );
        assert!(rn_eval(&inv, &Rational::from(0)).is_none());
    }

    #[test]
    fn polyk_div_rem_and_gcd() {
        // (k^2 - 1) = (k - 1)(k + 1)
        let k = PolyK::k();
        let one = PolyK::one();
        let a = k.sub(&one);
        let b = k.add(&one);
        let p = a.mul(&b);
        let (q, r) = PolyK::div_rem(&p, &a).expect("divide");
        assert!(r.is_zero());
        assert!(q.eq_poly(&b));
        let g = PolyK::gcd(&p, &b);
        assert!(g.eq_poly(&b.monic()));
    }

    #[test]
    fn polyk_shift_in_both_variables() {
        // p = k + n  ⇒  p(k+1) = k + n + 1, p with n↦n+1 = k + n + 1
        let p = PolyK::from_coeffs(vec![rn_var(), rn_one()]);
        let want = PolyK::from_coeffs(vec![rn_add(&rn_var(), &rn_one()), rn_one()]);
        assert!(p.shift_k(1).eq_poly(&want));
        assert!(p.shift_n(1).eq_poly(&want));
    }

    #[test]
    fn ratk_arithmetic_is_exact() {
        // 1/(k+n) + 1/(k-n) = 2k/(k^2 - n^2)
        let n = rn_var();
        let kp = PolyK::from_coeffs(vec![n.clone(), rn_one()]);
        let km = PolyK::from_coeffs(vec![rn_neg(&n), rn_one()]);
        let a = RatK::from_poly(kp.clone()).inv().unwrap();
        let b = RatK::from_poly(km.clone()).inv().unwrap();
        let s = a.add(&b);
        let want = RatK {
            num: PolyK::k().scale(&rn_int(2)),
            den: kp.mul(&km),
        }
        .normalize();
        assert!(s.eq_ratk(&want));
    }

    #[test]
    fn clear_denominators_makes_integer_primitive() {
        let n = rn_var();
        let half = rn_rat(Rational::from((1, 2)));
        let items = vec![rn_mul(&half, &n), rn_div(&rn_one(), &n).unwrap()];
        let out = clear_denominators(&items);
        assert_eq!(out.len(), 2);
        for p in &out {
            for c in &p.coeffs {
                assert_eq!(*c.clone().denom(), Integer::from(1));
            }
        }
    }
}
