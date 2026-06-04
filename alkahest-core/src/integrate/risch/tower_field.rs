//! Exponential tower differential field `ℚ(x)(t)`, `t = exp(η)`, as a
//! [`CoeffField`] — the Risch **MD** substrate for radicands that involve the
//! transcendental (e.g. `∛(x + eˣ)`, tutorial Example 15).
//!
//! An element is a rational function in `t` with `ℚ(x)`-coefficients
//! ([`RatFn`]), i.e. a fraction of polynomials in `t` over `ℚ(x)`.  The
//! polynomial-in-`t` arithmetic reuses the generic [`CoeffField`] machinery
//! instantiated at [`RationalFunctionField`] (the `ℚ(x)` coefficients), and the
//! derivation is the tower derivation
//!
//! ```text
//!   D(t) = η'·t,   D(Σⱼ cⱼ(x) tʲ) = Σⱼ (cⱼ'(x) + j·η'·cⱼ(x)) tʲ
//! ```
//!
//! Because `ℚ(x)(t)` is itself a [`CoeffField`], the generic `Quotient` gives
//! an algebraic extension `ℚ(x)(t)[y]/(q(x, t, y))` for free, and
//! `radical_dy` computes `D(y) = −q_x/q_y` in the tower.  This is exactly the
//! M0 "substitute a transcendental tower for the coefficient field" hook: the
//! radicand may now involve `t`.
//!
//! Scope: this provides the differential-algebra **substrate** (arithmetic +
//! derivation) for radicand-involving-transcendental.  The *integration* step —
//! solving the per-component Risch DE `vᵢ' + ωᵢ vᵢ = cᵢ` over this tower — is
//! the remaining MD work and is not done here.

use super::alg_field::{RatFn, RationalFunctionField};
use super::number_field::{
    gdegree, gext_gcd, gpoly_add, gpoly_divrem, gpoly_mul, gpoly_scale, gtrim, CoeffField, GPoly,
};

/// A polynomial in `t` over `ℚ(x)` (ascending degree in `t`).
type TPoly = GPoly<RationalFunctionField>;

fn qx() -> RationalFunctionField {
    RationalFunctionField
}

/// An element of `ℚ(x)(t)`: a canonical fraction `num/den` of `t`-polynomials
/// over `ℚ(x)` (coprime, monic `den`, `0 = ⟨⟩/1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TExpr {
    num: TPoly,
    den: TPoly,
}

impl TExpr {
    /// Build `num/den` in canonical form.  Panics if `den` is the zero
    /// polynomial.
    pub fn new(num: TPoly, den: TPoly) -> Self {
        let f = qx();
        let num = gtrim(&f, num);
        let den = gtrim(&f, den);
        assert!(!den.is_empty(), "TExpr: zero denominator");
        if num.is_empty() {
            return Self {
                num: Vec::new(),
                den: vec![f.one()],
            };
        }
        // Reduce by gcd over ℚ(x)[t].
        let (g, _, _) = gext_gcd(&f, &num, &den);
        let num = gpoly_divrem(&f, &num, &g).0;
        let den = gpoly_divrem(&f, &den, &g).0;
        // Normalize: make the denominator monic in t.
        let lead = den[gdegree(&f, &den) as usize].clone();
        let lead_inv = f
            .inv(&lead)
            .expect("nonzero ℚ(x) leading coeff is invertible");
        let num = gpoly_scale(&f, &num, &lead_inv);
        let den = gpoly_scale(&f, &den, &lead_inv);
        Self { num, den }
    }

    /// The constant (in `t`) element `r ∈ ℚ(x)`.
    pub fn from_ratfn(r: RatFn) -> Self {
        Self::new(vec![r], vec![qx().one()])
    }

    /// The monomial `t` (the exponential generator itself).
    pub fn t() -> Self {
        Self::new(vec![qx().zero(), qx().one()], vec![qx().one()])
    }

    /// Integer constant `n`.
    pub fn int(n: i64) -> Self {
        Self::from_ratfn(RatFn::int(n))
    }

    /// Numerator (canonical, polynomial in `t`).
    pub fn numer(&self) -> &TPoly {
        &self.num
    }
    /// Denominator (canonical, monic polynomial in `t`).
    pub fn denom(&self) -> &TPoly {
        &self.den
    }

    fn is_zero(&self) -> bool {
        self.num.is_empty()
    }

    fn add(&self, other: &Self) -> Self {
        let f = qx();
        let num = gpoly_add(
            &f,
            &gpoly_mul(&f, &self.num, &other.den),
            &gpoly_mul(&f, &other.num, &self.den),
        );
        Self::new(num, gpoly_mul(&f, &self.den, &other.den))
    }

    fn mul(&self, other: &Self) -> Self {
        let f = qx();
        Self::new(
            gpoly_mul(&f, &self.num, &other.num),
            gpoly_mul(&f, &self.den, &other.den),
        )
    }

    fn neg(&self) -> Self {
        let f = qx();
        let neg1 = f.neg(&f.one());
        Self::new(gpoly_scale(&f, &self.num, &neg1), self.den.clone())
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(Self::new(self.den.clone(), self.num.clone()))
        }
    }
}

/// `D(P)` for a `t`-polynomial `P = Σⱼ cⱼ tʲ`: `Σⱼ (cⱼ' + j·η'·cⱼ) tʲ`,
/// where `cⱼ' = d/dx cⱼ` and `η' = deta`.
fn tpoly_derivation(p: &TPoly, deta: &RatFn) -> TPoly {
    let f = qx();
    let mut out: TPoly = Vec::with_capacity(p.len());
    for (j, cj) in p.iter().enumerate() {
        let dc = f.derivation(cj); // cⱼ'
        let drift = f.mul(&f.mul(&RatFn::int(j as i64), deta), cj); // j·η'·cⱼ
        out.push(f.add(&dc, &drift));
    }
    gtrim(&f, out)
}

/// The exponential tower `ℚ(x)(t)`, `t = exp(η)`, parameterized by `η' = deta`.
#[derive(Clone, Debug)]
pub struct ExpTowerField {
    /// `η'(x)` — the derivative of the exponent, a `ℚ(x)` element.
    pub deta: RatFn,
}

impl ExpTowerField {
    pub fn new(deta: RatFn) -> Self {
        Self { deta }
    }
}

impl CoeffField for ExpTowerField {
    type Elem = TExpr;

    fn zero(&self) -> TExpr {
        TExpr::int(0)
    }
    fn one(&self) -> TExpr {
        TExpr::int(1)
    }
    fn from_i64(&self, n: i64) -> TExpr {
        TExpr::int(n)
    }
    fn add(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(b)
    }
    fn sub(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.add(&b.neg())
    }
    fn mul(&self, a: &TExpr, b: &TExpr) -> TExpr {
        a.mul(b)
    }
    fn neg(&self, a: &TExpr) -> TExpr {
        a.neg()
    }
    fn inv(&self, a: &TExpr) -> Option<TExpr> {
        a.inv()
    }
    fn is_zero(&self, a: &TExpr) -> bool {
        a.is_zero()
    }
    fn eq(&self, a: &TExpr, b: &TExpr) -> bool {
        a == b
    }

    /// `D(num/den) = (D(num)·den − num·D(den)) / den²`, with `D` the tower
    /// derivation on `t`-polynomials.
    fn derivation(&self, a: &TExpr) -> TExpr {
        let f = qx();
        let dnum = tpoly_derivation(&a.num, &self.deta);
        let dden = tpoly_derivation(&a.den, &self.deta);
        let numer = {
            let lhs = gpoly_mul(&f, &dnum, &a.den);
            let rhs = gpoly_mul(&f, &a.num, &dden);
            let neg1 = f.neg(&f.one());
            gpoly_add(&f, &lhs, &gpoly_scale(&f, &rhs, &neg1))
        };
        let denom = gpoly_mul(&f, &a.den, &a.den);
        TExpr::new(numer, denom)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::alg_field::{quotient_derivation, radical_dy};
    use crate::integrate::risch::number_field::Quotient;
    use rug::Rational;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// `x` as a `ℚ(x)(t)` constant-in-t element.
    fn x_elem() -> TExpr {
        TExpr::from_ratfn(RatFn::from_poly(&vec![rat(0), rat(1)]))
    }

    #[test]
    fn tower_derivation_of_t_is_eta_prime_t() {
        // η = x ⇒ η' = 1, t = eˣ, D(t) = t.
        let field = ExpTowerField::new(RatFn::int(1));
        let dt = field.derivation(&TExpr::t());
        assert_eq!(dt, TExpr::t());
        // D(x) = 1.
        assert_eq!(field.derivation(&x_elem()), TExpr::int(1));
        // D(x + t) = 1 + t.
        let x_plus_t = field.add(&x_elem(), &TExpr::t());
        let expected = field.add(&TExpr::int(1), &TExpr::t());
        assert_eq!(field.derivation(&x_plus_t), expected);
    }

    #[test]
    fn tower_quotient_rule() {
        // η = x.  D(1/t) = −η'·t / t² = −1/t.
        let field = ExpTowerField::new(RatFn::int(1));
        let inv_t = field.inv(&TExpr::t()).unwrap();
        let d = field.derivation(&inv_t);
        let neg_inv_t = field.neg(&inv_t);
        assert_eq!(d, neg_inv_t);
    }

    /// Build the radical extension `ℚ(x)(eˣ)[y]/(y³ − (x + eˣ))` and return
    /// `(quotient, dy)`.  This is the Example-15 field: a degree-3 radical whose
    /// radicand `x + eˣ` involves the transcendental.
    fn cbrt_x_plus_exp() -> (Quotient<ExpTowerField>, Vec<TExpr>) {
        let field = ExpTowerField::new(RatFn::int(1)); // t = eˣ
                                                       // modulus  y³ − (x + t):  [ −(x+t), 0, 0, 1 ].
        let neg_a = field.neg(&field.add(&x_elem(), &TExpr::t()));
        let modulus = vec![neg_a, field.zero(), field.zero(), field.one()];
        let q = Quotient::new(field, modulus);
        let dy = radical_dy(&q);
        (q, dy)
    }

    #[test]
    fn radical_over_exp_tower_derivation_consistency() {
        // In ℚ(x)(eˣ)[y]/(y³−(x+eˣ)):  D(y³) must equal D(x+eˣ) = 1 + eˣ.
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();

        // y³ reduces to the element (x + t).
        let one = field.one();
        let zero = field.zero();
        let y3 = q.reduce(&[zero.clone(), zero.clone(), zero, one]); // [0,0,0,1] mod q
        let x_plus_t = field.add(&x_elem(), &TExpr::t());
        assert!(
            q.elem_eq(&y3, std::slice::from_ref(&x_plus_t)),
            "y³ should reduce to x+eˣ"
        );

        // D(y³) via the extension derivation.
        let d_y3 = quotient_derivation(&q, &dy, &y3);
        let one_plus_t = field.add(&TExpr::int(1), &TExpr::t());
        assert!(
            q.elem_eq(&d_y3, &[one_plus_t]),
            "D(y³) = D(x+eˣ) = 1+eˣ; got {d_y3:?}"
        );
    }

    #[test]
    fn radical_over_exp_tower_leibniz() {
        // D(y·y²) = D(y)·y² + y·D(y²) in ℚ(x)(eˣ)(y).
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();
        let y = vec![field.zero(), field.one()]; // y
        let y2 = q.mul(&y, &y);

        let lhs = quotient_derivation(&q, &dy, &q.mul(&y, &y2));
        let rhs = q.add(
            &q.mul(&quotient_derivation(&q, &dy, &y), &y2),
            &q.mul(&y, &quotient_derivation(&q, &dy, &y2)),
        );
        assert!(
            q.elem_eq(&lhs, &rhs),
            "Leibniz over the tower: {lhs:?} vs {rhs:?}"
        );
    }

    #[test]
    fn radical_over_exp_tower_dy_value() {
        // D(y)·(3y²) = D(y³) = 1 + eˣ, i.e. 3y²·D(y) = 1+t.
        let (q, dy) = cbrt_x_plus_exp();
        let field = q.field().clone();
        let y = vec![field.zero(), field.one()];
        let y2 = q.mul(&y, &y);
        let three_y2 = q.mul(&[field.from_i64(3)], &y2);
        let prod = q.mul(&three_y2, &dy);
        let one_plus_t = field.add(&TExpr::int(1), &TExpr::t());
        assert!(
            q.elem_eq(&prod, &[one_plus_t]),
            "3y²·D(y) = 1+eˣ; got {prod:?}"
        );
    }
}
