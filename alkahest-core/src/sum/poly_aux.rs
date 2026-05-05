//! Helpers on [`RatUniPoly`] for affine substitution (shift) used by Gosper / Zeilberger-style summation.

use crate::matrix::normal_form::RatUniPoly;
use rug::Rational;

/// Substitute `x ↦ a·x + b` into polynomial `p` (so `p(a·x+b)`).
pub fn compose_affine(p: &RatUniPoly, a: &Rational, b: &Rational) -> RatUniPoly {
    if p.is_zero() {
        return RatUniPoly::zero();
    }
    let x = RatUniPoly::x();
    let axpb = &(&RatUniPoly::constant(a.clone()) * &x) + &RatUniPoly::constant(b.clone());
    let mut acc = RatUniPoly::zero();
    let mut pow = RatUniPoly::one();
    for coeff in &p.coeffs {
        acc = &acc + &(&pow * &RatUniPoly::constant(coeff.clone()));
        pow = &pow * &axpb;
    }
    acc.trim()
}

/// `gcd(p(x), q(x+j))` as polynomials in `x`.
pub fn gcd_shifted(p: &RatUniPoly, q: &RatUniPoly, j: i64) -> RatUniPoly {
    let qj = compose_affine(q, &Rational::from(1), &Rational::from(j));
    RatUniPoly::gcd(p, &qj)
}

pub fn poly_exact_div(a: &RatUniPoly, b: &RatUniPoly) -> Option<RatUniPoly> {
    if b.is_zero() {
        return None;
    }
    let (q, r) = RatUniPoly::div_rem(a, b);
    if r.is_zero() {
        Some(q)
    } else {
        None
    }
}

pub fn poly_nth_coeff(p: &RatUniPoly, n: i32) -> Rational {
    if n < 0 {
        return Rational::from(0);
    }
    let i = n as usize;
    if i >= p.coeffs.len() {
        Rational::from(0)
    } else {
        p.coeffs[i].clone()
    }
}
