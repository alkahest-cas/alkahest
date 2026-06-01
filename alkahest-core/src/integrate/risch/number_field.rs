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
//! generic `ℚ[x]`-modulo-a-polynomial operations (the modulus need not be
//! irreducible); they underpin both [`NumberField`] and the Hermite-reduction
//! step in [`super::rational_integrate`].

use rug::Rational;

use super::poly_rde::{degree, poly_add, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly};
use super::rational_rde::{poly_divrem, poly_sub};

// ---------------------------------------------------------------------------
// Base field: ℚ[x] modular arithmetic (modulus need not be irreducible)
// ---------------------------------------------------------------------------

/// Remainder of `a mod m` in `ℚ[x]`.
pub fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    poly_divrem(a, m).1
}

/// Extended GCD over `ℚ[x]`: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
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
// K-element constructors and K-polynomial-in-x arithmetic
// (used by the polynomial Risch DE over ℚ(α); see `poly_rde::solve_poly_rde_k`)
// ---------------------------------------------------------------------------

impl NumberField {
    /// The zero `K`-element.
    pub fn k_zero() -> KElem {
        Vec::new()
    }

    /// Embed an integer into `K`.
    pub fn from_int(&self, n: i64) -> KElem {
        self.reduce(&vec![Rational::from(n)])
    }

    /// Embed a rational into `K`.
    pub fn from_rational(&self, r: &Rational) -> KElem {
        self.reduce(&vec![r.clone()])
    }

    /// `a + b` of two `K`-polynomials in `x`.
    pub fn kpoly_add(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        let n = a.len().max(b.len());
        let mut r = vec![Self::k_zero(); n];
        for (i, c) in a.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        for (i, c) in b.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        Self::kpoly_trim(r)
    }

    /// `a − b` of two `K`-polynomials in `x`.
    pub fn kpoly_sub(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        let n = a.len().max(b.len());
        let mut r = vec![Self::k_zero(); n];
        for (i, c) in a.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        for (i, c) in b.iter().enumerate() {
            r[i] = self.sub(&r[i], c);
        }
        Self::kpoly_trim(r)
    }

    /// Scale a `K`-polynomial in `x` by a `K`-element.
    pub fn kpoly_scale(&self, p: &[KElem], s: &KElem) -> KPoly {
        if Self::is_zero(s) {
            return Vec::new();
        }
        Self::kpoly_trim(p.iter().map(|c| self.mul(c, s)).collect())
    }

    /// `a · b` of two `K`-polynomials in `x`.
    pub fn kpoly_mul(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        if Self::kdeg(a) < 0 || Self::kdeg(b) < 0 {
            return Vec::new();
        }
        let mut r = vec![Self::k_zero(); a.len() + b.len() - 1];
        for (i, ca) in a.iter().enumerate() {
            if Self::is_zero(ca) {
                continue;
            }
            for (j, cb) in b.iter().enumerate() {
                let p = self.mul(ca, cb);
                r[i + j] = self.add(&r[i + j], &p);
            }
        }
        Self::kpoly_trim(r)
    }

    /// `d/dx` of a `K`-polynomial in `x`.
    pub fn kpoly_deriv(&self, p: &[KElem]) -> KPoly {
        if p.len() <= 1 {
            return Vec::new();
        }
        Self::kpoly_trim(
            p[1..]
                .iter()
                .enumerate()
                .map(|(i, c)| self.mul(&self.from_int(i as i64 + 1), c))
                .collect(),
        )
    }

    /// `∫ dx` of a `K`-polynomial in `x` (constant of integration 0).
    pub fn kpoly_integrate(&self, p: &[KElem]) -> KPoly {
        let p = Self::kpoly_trim(p.to_vec());
        if p.is_empty() {
            return Vec::new();
        }
        let mut r = vec![Self::k_zero()]; // constant term 0
        for (i, c) in p.iter().enumerate() {
            let inv = self
                .inv(&self.from_int(i as i64 + 1))
                .expect("nonzero integer is invertible in a number field");
            r.push(self.mul(c, &inv));
        }
        Self::kpoly_trim(r)
    }

    /// `p^n` of a `K`-polynomial in `x` (`n ≥ 0`; `p^0 = 1`).
    pub fn kpoly_pow(&self, p: &[KElem], n: u32) -> KPoly {
        let mut acc = vec![self.from_int(1)];
        for _ in 0..n {
            acc = self.kpoly_mul(&acc, p);
        }
        acc
    }

    /// Make a `K`-polynomial monic in `x` (divide by its leading coefficient).
    /// The zero polynomial is returned unchanged; `None` if the leading
    /// coefficient is not invertible in `K`.
    pub fn kpoly_monic(&self, p: &[KElem]) -> Option<KPoly> {
        let d = Self::kdeg(p);
        if d < 0 {
            return Some(Vec::new());
        }
        let inv = self.inv(&p[d as usize])?;
        Some(Self::kpoly_trim(
            p.iter().map(|c| self.mul(c, &inv)).collect(),
        ))
    }

    /// Exact division `a / b` of `K`-polynomials in `x`; `None` if `b` does not
    /// divide `a` evenly (or `b` is zero / has a non-invertible leading term).
    pub fn kpoly_div_exact(&self, a: &[KElem], b: &[KElem]) -> Option<KPoly> {
        let (q, r) = self.kpoly_divrem(a, b)?;
        if Self::kdeg(&r) >= 0 {
            return None; // nonzero remainder
        }
        Some(q)
    }

    /// Coefficient of `x^i` in a `K`-polynomial; the zero element for `i < 0` or
    /// out of range.
    pub fn kcoeff(p: &[KElem], i: i64) -> KElem {
        if i < 0 {
            return Self::k_zero();
        }
        p.get(i as usize).cloned().unwrap_or_else(Self::k_zero)
    }

    /// Equality of two `K`-polynomials in `x` (coefficient-wise after trimming).
    pub fn kpoly_eq(a: &[KElem], b: &[KElem]) -> bool {
        let a = Self::kpoly_trim(a.to_vec());
        let b = Self::kpoly_trim(b.to_vec());
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| trim(x.clone()) == trim(y.clone()))
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

    #[test]
    fn kpoly_mul_over_sqrt2() {
        // (x + √2)·(x − √2) = x² − 2  over ℚ(√2).
        let k = q_sqrt2();
        let t = vec![rat(0), rat(1)]; // √2
        let a = vec![t.clone(), vec![rat(1)]]; // √2 + x
        let b = vec![k.neg(&t), vec![rat(1)]]; // −√2 + x
        let p = k.kpoly_mul(&a, &b);
        assert_eq!(NumberField::kdeg(&p), 2);
        assert_eq!(trim(p[0].clone()), vec![rat(-2)]); // constant −2 (√2·−√2)
        assert!(NumberField::is_zero(&p[1])); // x coefficient cancels
        assert_eq!(trim(p[2].clone()), vec![rat(1)]); // x²
    }

    #[test]
    fn kpoly_deriv_integrate_roundtrip() {
        // d/dx ∫ (√2·x² + x) dx == √2·x² + x.
        let k = q_sqrt2();
        let t = vec![rat(0), rat(1)];
        let p = vec![NumberField::k_zero(), vec![rat(1)], t.clone()]; // x + √2·x²
        let integ = k.kpoly_integrate(&p);
        let back = k.kpoly_deriv(&integ);
        // Compare coefficient-wise (trimmed).
        let p_t = NumberField::kpoly_trim(p);
        let back_t = NumberField::kpoly_trim(back);
        assert_eq!(p_t.len(), back_t.len());
        for (a, b) in p_t.iter().zip(back_t.iter()) {
            assert_eq!(trim(a.clone()), trim(b.clone()));
        }
    }
}
