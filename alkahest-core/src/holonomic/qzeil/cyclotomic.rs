//! Exact arithmetic in the cyclotomic field `Q(ζ_d) = Q[q]/(Φ_d(q))`, and the
//! `Φ_d`-adic valuation of an element of `Q(q)`.
//!
//! This is the arithmetic layer under [`super::rootofunity`]. Everything here
//! is exact: `Φ_d` is built by exact polynomial division from `q^d − 1`, the
//! residue ring is `Q[q]` reduced by `Φ_d`, and inversion is the extended
//! Euclidean algorithm. **Nothing is evaluated numerically** — a "does this
//! vanish at `ζ_d`" question is answered by polynomial divisibility over `Q`,
//! never by plugging in a floating-point root of unity.
//!
//! # Why divisibility is the right test
//!
//! `Φ_d` is irreducible over `Q` (a classical theorem), so for `p ∈ Q[q]`
//!
//! ```text
//! p(ζ_d) = 0   ⟺   Φ_d | p
//! ```
//!
//! and the quotient `Q[q]/(Φ_d)` is a *field*, which is what makes [`inv`]
//! total on non-zero elements. Both facts are used below and neither is
//! approximated.
//!
//! # The valuation, and why a caller wants it
//!
//! Every `r ∈ Q(q)` factors as `r = Φ_d^v · (N/D)` with `Φ_d ∤ N`, `Φ_d ∤ D`.
//! That `v = ` [`CycloField::valuation`] is the exact statement "`Φ_d(q)^v`
//! divides `r`", which is the shape a `q`-supercongruence takes. `v ≥ 0` is
//! exactly the condition for `r` to have a value at `ζ_d` at all, so the same
//! computation decides specialisability and measures divisibility.
//!
//! [`inv`]: CycloField::inv

use crate::matrix::normal_form::RatUniPoly;
use rug::Rational;
use std::collections::BTreeMap;

/// Largest order of a root of unity this module will build a field for.
///
/// The cost of the residue-ring arithmetic grows as `φ(d)²`, so the cap is a
/// resource bound rather than a mathematical one.
pub const MAX_CYCLOTOMIC_ORDER: u32 = 512;

/// `q^e − 1`.
fn q_pow_minus_one(e: u32) -> RatUniPoly {
    let mut coeffs = vec![Rational::from(0); e as usize + 1];
    coeffs[0] = Rational::from(-1);
    coeffs[e as usize] = Rational::from(1);
    RatUniPoly { coeffs }
}

/// Exact division, or `None` when `b` does not divide `a`.
fn exact_div(a: &RatUniPoly, b: &RatUniPoly) -> Option<RatUniPoly> {
    if b.is_zero() {
        return None;
    }
    let (quo, rem) = RatUniPoly::div_rem(a, b);
    rem.is_zero().then_some(quo)
}

/// The `d`-th cyclotomic polynomial `Φ_d(q) ∈ Z[q]`, monic of degree `φ(d)`.
///
/// Built from `q^d − 1 = ∏_{e | d} Φ_e(q)` by exact division, with the proper
/// divisors memoised so the work is one division per divisor rather than one
/// per divisor *chain*.
///
/// Returns `Φ_1 = q − 1` at `d = 1`; that is the honest answer — "the primitive
/// first root of unity" is `1`, and specialising there is the classical `q → 1`
/// limit.
pub fn cyclotomic_polynomial(d: u32) -> RatUniPoly {
    debug_assert!(d >= 1, "the order of a root of unity is at least 1");
    let d = d.max(1);
    let mut memo: BTreeMap<u32, RatUniPoly> = BTreeMap::new();
    for e in 1..=d {
        if d % e != 0 {
            continue;
        }
        let mut num = q_pow_minus_one(e);
        for (&f, phi_f) in memo.iter() {
            if e % f == 0 {
                // `Φ_f` divides `q^e − 1` whenever `f | e`, so the division is
                // exact; the fallback keeps this total rather than panicking.
                num = exact_div(&num, phi_f).unwrap_or(num);
            }
        }
        memo.insert(e, num);
    }
    memo.remove(&d).unwrap_or_else(RatUniPoly::one)
}

/// An element of `Q(ζ_d)`, as its canonical representative of degree `< φ(d)`.
///
/// Two elements are equal exactly when their representatives are, which is what
/// makes `==` on this type an exact decision in the cyclotomic field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycloElem {
    /// The canonical representative in `Q[q]`, of degree `< φ(d)`.
    pub poly: RatUniPoly,
}

impl CycloElem {
    /// Whether this is `0` in `Q(ζ_d)`.
    pub fn is_zero(&self) -> bool {
        self.poly.is_zero()
    }
}

/// The field `Q(ζ_d) = Q[q]/(Φ_d(q))`, and the `Φ_d`-adic valuation on `Q(q)`.
#[derive(Clone, Debug)]
pub struct CycloField {
    d: u32,
    phi: RatUniPoly,
}

impl CycloField {
    /// The field of `d`-th roots of unity, or `None` past
    /// [`MAX_CYCLOTOMIC_ORDER`] (or at `d = 0`, which names nothing).
    pub fn new(d: u32) -> Option<CycloField> {
        if d == 0 || d > MAX_CYCLOTOMIC_ORDER {
            return None;
        }
        Some(CycloField {
            d,
            phi: cyclotomic_polynomial(d),
        })
    }

    /// The order `d` of the root of unity.
    pub fn order(&self) -> u32 {
        self.d
    }

    /// `Φ_d(q)`.
    pub fn modulus(&self) -> &RatUniPoly {
        &self.phi
    }

    /// `φ(d)` — the degree of the extension `Q(ζ_d) / Q`.
    pub fn degree(&self) -> usize {
        self.phi.degree().max(0) as usize
    }

    /// The canonical representative of `p mod Φ_d`.
    pub fn reduce(&self, p: &RatUniPoly) -> CycloElem {
        let (_, rem) = RatUniPoly::div_rem(p, &self.phi);
        CycloElem { poly: rem.trim() }
    }

    /// `0`.
    pub fn zero(&self) -> CycloElem {
        CycloElem {
            poly: RatUniPoly::zero(),
        }
    }

    /// `1`.
    pub fn one(&self) -> CycloElem {
        self.reduce(&RatUniPoly::one())
    }

    /// A rational constant.
    pub fn from_rational(&self, c: Rational) -> CycloElem {
        self.reduce(&RatUniPoly::constant(c))
    }

    /// `ζ_d^e`, for any sign of `e` — `ζ_d^d = 1`, so the exponent is reduced
    /// modulo `d` first and no inversion is needed.
    pub fn zeta_pow(&self, e: i64) -> CycloElem {
        let r = e.rem_euclid(self.d as i64) as usize;
        let mut coeffs = vec![Rational::from(0); r + 1];
        coeffs[r] = Rational::from(1);
        self.reduce(&RatUniPoly { coeffs })
    }

    /// `a + b`.
    pub fn add(&self, a: &CycloElem, b: &CycloElem) -> CycloElem {
        self.reduce(&(&a.poly + &b.poly))
    }

    /// `a − b`.
    pub fn sub(&self, a: &CycloElem, b: &CycloElem) -> CycloElem {
        self.reduce(&(&a.poly - &b.poly))
    }

    /// `−a`.
    pub fn neg(&self, a: &CycloElem) -> CycloElem {
        CycloElem { poly: -&a.poly }
    }

    /// `a · b`.
    pub fn mul(&self, a: &CycloElem, b: &CycloElem) -> CycloElem {
        if a.is_zero() || b.is_zero() {
            return self.zero();
        }
        self.reduce(&(&a.poly * &b.poly))
    }

    /// `a⁻¹`, or `None` at `a = 0`.
    ///
    /// Total on non-zero elements because `Φ_d` is irreducible over `Q`: the
    /// extended Euclidean algorithm returns `gcd(a, Φ_d) = 1` for every `a` the
    /// modulus does not divide, and a reduced representative is never divisible
    /// by `Φ_d` unless it is zero.
    pub fn inv(&self, a: &CycloElem) -> Option<CycloElem> {
        if a.is_zero() {
            return None;
        }
        let (s, _, g) = RatUniPoly::gcdex(&a.poly, &self.phi);
        if g.degree() != 0 {
            // Unreachable for an irreducible modulus; refuse rather than
            // return a wrong inverse if it ever happens.
            return None;
        }
        let c = g.coeffs.first()?.clone();
        if c == 0 {
            return None;
        }
        let scaled = &s * &RatUniPoly::constant(Rational::from(1) / c);
        Some(self.reduce(&scaled))
    }

    /// `Φ_d`-adic valuation of a polynomial, with the cofactor: the unique
    /// `(v, p')` with `p = Φ_d^v · p'` and `Φ_d ∤ p'`. `None` at `p = 0`.
    fn poly_valuation(&self, p: &RatUniPoly) -> Option<(i64, RatUniPoly)> {
        if p.is_zero() {
            return None;
        }
        let mut v = 0_i64;
        let mut cur = p.clone();
        while let Some(next) = exact_div(&cur, &self.phi) {
            if next.is_zero() {
                break;
            }
            cur = next;
            v += 1;
        }
        Some((v, cur))
    }

    /// The exact `Φ_d`-adic valuation of `r ∈ Q(q)`: the integer `v` with
    /// `r = Φ_d^v · N/D` and `Φ_d` dividing neither `N` nor `D`.
    ///
    /// `None` means `r = 0`, whose valuation is `+∞`. `v ≥ 0` is exactly the
    /// condition for [`specialize`](Self::specialize) to succeed, and `v ≥ 1`
    /// is exactly the divisibility `Φ_d(q)^v | r` a `q`-supercongruence asserts.
    pub fn valuation(&self, r: &crate::holonomic::qfield::Rn) -> Option<i64> {
        let (vn, _) = self.poly_valuation(&r.num)?;
        let (vd, _) = self
            .poly_valuation(&r.den)
            .unwrap_or((0, RatUniPoly::one()));
        Some(vn - vd)
    }

    /// `r(ζ_d)` as an element of `Q(ζ_d)`, or `None` when `r` has a **pole**
    /// there — i.e. exactly when [`valuation`](Self::valuation) is negative.
    ///
    /// The numerator and denominator are stripped of their `Φ_d` factors first,
    /// so this is correct even if the caller hands over a representation that
    /// is not in lowest terms: `(q^d − 1)/(q^d − 1)` specialises to `1`, not to
    /// `0/0`.
    pub fn specialize(&self, r: &crate::holonomic::qfield::Rn) -> Option<CycloElem> {
        let Some((vn, num)) = self.poly_valuation(&r.num) else {
            return Some(self.zero());
        };
        let (vd, den) = self
            .poly_valuation(&r.den)
            .unwrap_or((0, RatUniPoly::one()));
        match (vn - vd).cmp(&0) {
            std::cmp::Ordering::Less => None,
            std::cmp::Ordering::Greater => Some(self.zero()),
            std::cmp::Ordering::Equal => {
                let n = self.reduce(&num);
                let d = self.reduce(&den);
                let inv = self.inv(&d)?;
                Some(self.mul(&n, &inv))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::qfield::{rn_one, rn_poly, Rn};

    fn poly(coeffs: &[i64]) -> RatUniPoly {
        RatUniPoly {
            coeffs: coeffs.iter().map(|&c| Rational::from(c)).collect(),
        }
        .trim()
    }

    #[test]
    fn cyclotomic_polynomials_match_the_classical_table() {
        assert_eq!(cyclotomic_polynomial(1), poly(&[-1, 1])); // q − 1
        assert_eq!(cyclotomic_polynomial(2), poly(&[1, 1])); // q + 1
        assert_eq!(cyclotomic_polynomial(3), poly(&[1, 1, 1]));
        assert_eq!(cyclotomic_polynomial(4), poly(&[1, 0, 1]));
        assert_eq!(cyclotomic_polynomial(5), poly(&[1, 1, 1, 1, 1]));
        assert_eq!(cyclotomic_polynomial(6), poly(&[1, -1, 1]));
        assert_eq!(cyclotomic_polynomial(12), poly(&[1, 0, -1, 0, 1]));
        // The first cyclotomic polynomial with a coefficient outside {−1,0,1}.
        assert_eq!(cyclotomic_polynomial(105).coeffs[7], Rational::from(-2));
    }

    #[test]
    fn the_product_over_divisors_is_q_to_the_d_minus_one() {
        for d in 1_u32..=24 {
            let mut prod = RatUniPoly::one();
            for e in 1..=d {
                if d % e == 0 {
                    prod = &prod * &cyclotomic_polynomial(e);
                }
            }
            assert_eq!(
                prod,
                q_pow_minus_one(d),
                "prod_(e|{d}) Phi_e must be q^{d} − 1"
            );
        }
    }

    #[test]
    fn degree_is_eulers_totient() {
        let totient = |d: u32| (1..=d).filter(|&e| gcd(e, d) == 1).count();
        for d in 1_u32..=40 {
            let f = CycloField::new(d).expect("in range");
            assert_eq!(f.degree(), totient(d), "deg Phi_{d} must be phi({d})");
        }
    }

    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    #[test]
    fn zeta_has_order_exactly_d() {
        for d in 1_u32..=12 {
            let f = CycloField::new(d).expect("in range");
            assert_eq!(f.zeta_pow(d as i64), f.one(), "zeta^{d} must be 1");
            assert_eq!(f.zeta_pow(-1), f.zeta_pow(d as i64 - 1));
            for e in 1..d {
                // A primitive d-th root of unity: no smaller power is 1.
                assert_ne!(f.zeta_pow(e as i64), f.one(), "zeta^{e} must not be 1");
            }
        }
    }

    #[test]
    fn inverse_is_a_two_sided_inverse() {
        for d in [1_u32, 2, 3, 4, 5, 6, 8, 12] {
            let f = CycloField::new(d).expect("in range");
            for e in 0..d as i64 {
                let a = f.add(&f.zeta_pow(e), &f.from_rational(Rational::from(3)));
                if a.is_zero() {
                    continue;
                }
                let inv = f.inv(&a).expect("nonzero elements are invertible");
                assert_eq!(f.mul(&a, &inv), f.one());
            }
        }
    }

    #[test]
    fn valuation_counts_the_phi_factors_exactly() {
        let f = CycloField::new(3).expect("in range");
        // (q³ − 1) = (q − 1)·Φ_3, so v = 1.
        let r: Rn = rn_poly(q_pow_minus_one(3));
        assert_eq!(f.valuation(&r), Some(1));
        // Φ_3² has valuation 2.
        let phi2 = &cyclotomic_polynomial(3) * &cyclotomic_polynomial(3);
        assert_eq!(f.valuation(&rn_poly(phi2)), Some(2));
        // 1/(q³ − 1) has valuation −1 and no value at ζ_3.
        let inv = crate::holonomic::qfield::rn_inv(&r).expect("nonzero");
        assert_eq!(f.valuation(&inv), Some(-1));
        assert!(f.specialize(&inv).is_none(), "a pole must not specialise");
        // 0 has no valuation.
        assert_eq!(f.valuation(&crate::holonomic::qfield::rn_zero()), None);
        assert_eq!(f.valuation(&rn_one()), Some(0));
    }

    #[test]
    fn a_zero_over_zero_representation_still_specialises() {
        // `Rn` is kept reduced, but `specialize` must not depend on that: it
        // strips the Φ factors from both sides first.
        let f = CycloField::new(3).expect("in range");
        let cube = q_pow_minus_one(3);
        let r = Rn {
            num: cube.clone(),
            den: cube,
        };
        assert_eq!(f.specialize(&r), Some(f.one()));
        assert_eq!(f.valuation(&r), Some(0));
    }

    #[test]
    fn specialising_a_root_of_unity_power_agrees_with_zeta_pow() {
        for d in 1_u32..=8 {
            let f = CycloField::new(d).expect("in range");
            for e in 0..12_i64 {
                let r = super::super::field::qq_pow(e);
                assert_eq!(f.specialize(&r), Some(f.zeta_pow(e)));
            }
        }
    }

    #[test]
    fn out_of_range_orders_are_refused() {
        assert!(CycloField::new(0).is_none());
        assert!(CycloField::new(MAX_CYCLOTOMIC_ORDER + 1).is_none());
        assert!(CycloField::new(MAX_CYCLOTOMIC_ORDER).is_some());
    }
}
