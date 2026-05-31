//! Algebraic number field `K = ℚ[t]/(q(t))` arithmetic.
//!
//! An element of `K` is represented as a [`QPoly`] of degree `< deg(q)` — a
//! ℚ-polynomial in the field generator `t`, reduced modulo the minimal
//! polynomial `q`.  A polynomial in a second variable `x` with coefficients in
//! `K` is a [`KPoly`] (`Vec<KElem>`, ascending degree in `x`).
//!
//! This module concentrates the quotient-ring primitives that were previously
//! private to [`super::rational_integrate`] (where they were introduced for the
//! Lazard–Rioboo–Trager degree-≥3 `RootSum` log argument, computing
//! `gcd_x(N − t·P', P)` in `ℚ(c)`).  They are factored out here so the same
//! arithmetic can back the **ℚ(α)-coefficient RDE work** (Risch Gap E): the
//! polynomial and rational Risch DE solvers currently operate over `ℚ` only;
//! threading a [`NumberField`] through them is what lets `√2·exp(x)`-style
//! algebraic-number *coefficients* (as opposed to constant factors) be handled.
//!
//! The three free functions [`poly_mod`], [`ext_gcd`], and [`mod_inverse`] are
//! generic ℚ[x]-modulo-a-polynomial operations (the modulus need not be
//! irreducible); they underpin both [`NumberField`] and the Hermite-reduction
//! step in [`super::rational_integrate`].

use rug::Rational;

use super::poly_rde::{degree, poly_add, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly};
use super::rational_rde::{poly_divrem, poly_sub};

// ---------------------------------------------------------------------------
// Base field: ℚ[x] modular arithmetic (modulus need not be irreducible)
// ---------------------------------------------------------------------------

/// Remainder of `a mod m` in ℚ[x].
pub fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    poly_divrem(a, m).1
}

/// Extended GCD over ℚ[x]: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
pub fn ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    let (mut old_r, mut r) = (trim(a.clone()), trim(b.clone()));
    let (mut old_s, mut s) = (poly_one(), poly_zero());
    let (mut old_t, mut t) = (poly_zero(), poly_one());
    while !r.is_empty() {
        let (q, rem) = poly_divrem(&old_r, &r);
        old_r = r;
        r = rem;
        let ns = poly_sub(&old_s, &poly_mul(&q, &s));
        old_s = s;
        s = ns;
        let nt = poly_sub(&old_t, &poly_mul(&q, &t));
        old_t = t;
        t = nt;
    }
    let dg = degree(&old_r);
    if dg < 0 {
        return (poly_zero(), old_s, old_t);
    }
    let inv = Rational::from(1) / old_r[dg as usize].clone();
    (
        poly_scale(&old_r, &inv),
        poly_scale(&old_s, &inv),
        poly_scale(&old_t, &inv),
    )
}

/// Inverse of `w` modulo `v` (requires `gcd(w, v) = 1`), else `None`.
pub fn mod_inverse(w: &QPoly, v: &QPoly) -> Option<QPoly> {
    let (g, s, _t) = ext_gcd(w, v);
    if degree(&g) != 0 {
        return None; // not coprime
    }
    Some(poly_mod(&s, v))
}

// ---------------------------------------------------------------------------
// Number field K = ℚ[t]/(q)
// ---------------------------------------------------------------------------

/// An element of `K` — a ℚ-polynomial in the generator `t`, reduced mod `q`.
pub type KElem = QPoly;

/// A polynomial in `x` over `K` (ascending degree in `x`; each coefficient is a
/// [`KElem`]).
pub type KPoly = Vec<KElem>;

/// A simple algebraic number field `K = ℚ[t]/(q(t))`.
///
/// `q` should be squarefree (irreducible for a genuine field; squarefree is
/// enough for the residue-by-residue Lazard–Rioboo–Trager use, where any
/// non-invertible leading coefficient simply makes an operation return `None`).
#[derive(Clone, Debug)]
pub struct NumberField {
    modulus: QPoly,
}

impl NumberField {
    /// Build `K = ℚ[t]/(modulus)`.
    pub fn new(modulus: QPoly) -> Self {
        Self {
            modulus: trim(modulus),
        }
    }

    /// The defining (minimal) polynomial `q(t)`.
    pub fn modulus(&self) -> &QPoly {
        &self.modulus
    }

    /// Field degree `[K : ℚ] = deg q`.
    pub fn degree(&self) -> i64 {
        degree(&self.modulus)
    }

    /// Reduce an arbitrary ℚ-polynomial in `t` into canonical `K`-element form.
    pub fn reduce(&self, a: &KElem) -> KElem {
        poly_mod(a, &self.modulus)
    }

    /// `a + b` in `K`.
    pub fn add(&self, a: &KElem, b: &KElem) -> KElem {
        self.reduce(&poly_add(a, b))
    }

    /// `a − b` in `K`.
    pub fn sub(&self, a: &KElem, b: &KElem) -> KElem {
        self.reduce(&poly_sub(a, b))
    }

    /// `a · b` in `K`.
    pub fn mul(&self, a: &KElem, b: &KElem) -> KElem {
        self.reduce(&poly_mul(a, b))
    }

    /// `−a` in `K`.
    pub fn neg(&self, a: &KElem) -> KElem {
        self.reduce(&poly_scale(a, &Rational::from(-1)))
    }

    /// Multiplicative inverse `a⁻¹` in `K`, or `None` when `a` is a zero divisor
    /// (e.g. shares a factor with a non-irreducible modulus) or zero.
    pub fn inv(&self, a: &KElem) -> Option<KElem> {
        mod_inverse(a, &self.modulus)
    }

    /// Is the `K`-element zero?
    pub fn is_zero(a: &KElem) -> bool {
        trim(a.clone()).is_empty()
    }

    // -- polynomials in x over K --

    /// Degree (in `x`) of a `K`-polynomial; `-1` for the zero polynomial.
    pub fn kdeg(p: &[KElem]) -> i64 {
        let mut d = p.len() as i64 - 1;
        while d >= 0 && Self::is_zero(&p[d as usize]) {
            d -= 1;
        }
        d
    }

    /// Drop trailing zero `K`-coefficients.
    pub fn kpoly_trim(mut p: KPoly) -> KPoly {
        while p.last().is_some_and(Self::is_zero) {
            p.pop();
        }
        p
    }

    /// Euclidean division of `K`-polynomials in `x`; returns `(quot, rem)` with
    /// `a = quot·b + rem` and `deg_x rem < deg_x b`.  `None` if `b` is zero or a
    /// leading coefficient is not invertible in `K`.
    pub fn kpoly_divrem(&self, a: &[KElem], b: &[KElem]) -> Option<(KPoly, KPoly)> {
        let bd = Self::kdeg(b);
        if bd < 0 {
            return None; // division by zero
        }
        let lead_inv = self.inv(&b[bd as usize])?;
        let mut r = Self::kpoly_trim(a.to_vec());
        let ad = Self::kdeg(&r);
        if ad < bd {
            return Some((vec![], r));
        }
        let mut quot = vec![poly_zero(); (ad - bd + 1) as usize];
        loop {
            let rd = Self::kdeg(&r);
            if rd < bd {
                break;
            }
            let coeff = self.mul(&r[rd as usize], &lead_inv);
            let shift = (rd - bd) as usize;
            for (i, bi) in b.iter().enumerate() {
                if shift + i < r.len() {
                    r[shift + i] = self.sub(&r[shift + i], &self.mul(&coeff, bi));
                }
            }
            quot[shift] = coeff;
            r = Self::kpoly_trim(r);
            if r.is_empty() {
                break;
            }
        }
        Some((Self::kpoly_trim(quot), r))
    }

    /// Monic (in `x`) GCD of two `K`-polynomials.  `None` if both are zero or a
    /// leading coefficient is not invertible in `K`.
    pub fn kpoly_gcd(&self, a: &[KElem], b: &[KElem]) -> Option<KPoly> {
        let mut a = Self::kpoly_trim(a.to_vec());
        let mut b = Self::kpoly_trim(b.to_vec());
        while Self::kdeg(&b) >= 0 {
            let (_, rem) = self.kpoly_divrem(&a, &b)?;
            a = b;
            b = rem;
        }
        let ad = Self::kdeg(&a);
        if ad < 0 {
            return None;
        }
        // Make monic in x.
        let lead_inv = self.inv(&a[ad as usize])?;
        Some(a.iter().map(|c| self.mul(c, &lead_inv)).collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }
    fn frac(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    /// K = ℚ(√2) = ℚ[t]/(t²−2).
    fn q_sqrt2() -> NumberField {
        NumberField::new(vec![rat(-2), rat(0), rat(1)])
    }

    #[test]
    fn base_field_mod_inverse() {
        // In ℚ[x]/(x²+1), x·x = −1, so x⁻¹ = −x.
        let m = vec![rat(1), rat(0), rat(1)]; // x²+1
        let x = vec![rat(0), rat(1)];
        let inv = mod_inverse(&x, &m).expect("x is a unit mod x²+1");
        // x · inv ≡ 1.
        assert_eq!(trim(poly_mod(&poly_mul(&x, &inv), &m)), vec![rat(1)]);
    }

    #[test]
    fn sqrt2_arithmetic() {
        let k = q_sqrt2();
        let t = vec![rat(0), rat(1)]; // √2
                                      // (1+√2)² = 3 + 2√2.
        let one_plus = k.add(&vec![rat(1)], &t);
        let sq = k.mul(&one_plus, &one_plus);
        assert_eq!(trim(sq), vec![rat(3), rat(2)]);
        // (√2)⁻¹ = √2/2.
        let inv = k.inv(&t).expect("√2 invertible");
        assert_eq!(trim(inv.clone()), vec![rat(0), frac(1, 2)]);
        assert_eq!(trim(k.mul(&t, &inv)), vec![rat(1)]);
    }

    #[test]
    fn kpoly_gcd_splits_over_sqrt2() {
        // Over ℚ(√2), gcd_x(x²−2, x−√2) = x−√2.
        let k = q_sqrt2();
        let neg_t = k.neg(&vec![rat(0), rat(1)]); // −√2
                                                  // a = x²−2  (coeffs in K, constants): [−2, 0, 1].
        let a: KPoly = vec![vec![rat(-2)], poly_zero(), vec![rat(1)]];
        // b = x−√2: [−√2, 1].
        let b: KPoly = vec![neg_t.clone(), vec![rat(1)]];
        let g = k.kpoly_gcd(&a, &b).expect("gcd exists");
        // Monic in x: [−√2, 1].
        assert_eq!(NumberField::kdeg(&g), 1);
        assert_eq!(trim(g[0].clone()), trim(neg_t));
        assert_eq!(trim(g[1].clone()), vec![rat(1)]);
    }

    #[test]
    fn non_unit_leading_coeff_is_none() {
        // In ℚ[t]/(t²−1) (not a field), t−1 is a zero divisor → no inverse.
        let k = NumberField::new(vec![rat(-1), rat(0), rat(1)]);
        assert!(k.inv(&vec![rat(-1), rat(1)]).is_none());
    }
}
