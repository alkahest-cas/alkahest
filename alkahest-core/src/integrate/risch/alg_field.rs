//! `x`-dependent algebraic *function* field extension `ℚ(x)[y]/(q(x, y))`
//! carrying a derivation — Risch milestone **M0**.
//!
//! Where [`super::number_field`] models a *constant* algebraic extension
//! `ℚ[t]/(q)` (the coefficient field `ℚ` has zero derivation), this module
//! models an algebraic extension of the **rational-function field** `ℚ(x)`,
//! whose generator `y` satisfies `q(x, y) = 0` and therefore has a non-trivial
//! derivative `D(y) = −q_x / q_y` forced by differentiating that relation.
//! This is the substrate the mixed algebraic + transcendental integrator (MA,
//! M1) builds on: it generalizes the rank-2 `KPair` (`α² = p(x)`,
//! `α' = (p'/2p)·α`) in [`super::exp_case`] to **arbitrary degree `d`**.
//!
//! ## How it reuses the generic core
//!
//! The whole point of the M0 refactor in [`super::number_field`] was that the
//! polynomial-quotient arithmetic is generic over a [`CoeffField`].  Here the
//! coefficient field is [`RationalFunctionField`] (`ℚ(x)`), and the extension
//! ring is exactly [`Quotient`]`<`[`RationalFunctionField`]`>` — no quotient
//! arithmetic is re-implemented.  The genuinely new primitive this module adds
//! is the **derivation** `D` on that ring (everything else is inherited).
//!
//! The derivation of a general element `a = Σⱼ bⱼ(x) yʲ` is
//!
//! ```text
//!   D(a) = Σⱼ bⱼ'(x) yʲ  +  (∂a/∂y) · D(y)            (reduced mod q)
//! ```
//!
//! — the first sum is the coefficient-field derivation applied coefficient-wise
//! (the "explicit `x`" part); the second is the chain-rule term through the
//! algebraic generator, where `D(y) = −q_x/q_y` is computed once at
//! construction.

use rug::Rational;

use super::number_field::{CoeffField, GPoly, Quotient};
use super::poly_rde::{degree, poly_deriv, poly_mul, poly_one, poly_scale, poly_zero, trim, QPoly};
use super::rational_rde::{poly_div_exact, poly_gcd, poly_sub};

// ===========================================================================
// ℚ(x) — the rational-function coefficient field
// ===========================================================================

/// An element of `ℚ(x)`: a rational function `num/den`, kept in canonical form
/// (coprime `num`/`den`, monic `den`, `0 = ⟨⟩/1`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RatFn {
    num: QPoly,
    den: QPoly,
}

impl RatFn {
    /// Build `num/den` in canonical form.  Panics if `den` is the zero
    /// polynomial.
    pub fn new(num: QPoly, den: QPoly) -> Self {
        let num = trim(num);
        let den = trim(den);
        assert!(!den.is_empty(), "RatFn: zero denominator");
        if num.is_empty() {
            return Self {
                num: Vec::new(),
                den: poly_one(),
            };
        }
        // Reduce by gcd.
        let g = poly_gcd(&num, &den);
        let mut num = poly_div_exact(&num, &g);
        let mut den = poly_div_exact(&den, &g);
        // Normalize so the denominator is monic.
        let lc = den[degree(&den) as usize].clone();
        if lc != 1 {
            let inv = Rational::from(1) / lc;
            num = poly_scale(&num, &inv);
            den = poly_scale(&den, &inv);
        }
        Self { num, den }
    }

    /// A rational function from a polynomial `p(x)` (denominator 1).
    pub fn from_poly(p: &QPoly) -> Self {
        Self::new(p.clone(), poly_one())
    }

    /// The integer constant `n`.
    pub fn int(n: i64) -> Self {
        Self::new(vec![Rational::from(n)], poly_one())
    }

    /// The numerator (canonical form).
    pub fn numer(&self) -> &QPoly {
        &self.num
    }

    /// The denominator (canonical form, monic).
    pub fn denom(&self) -> &QPoly {
        &self.den
    }

    fn is_zero(&self) -> bool {
        self.num.is_empty()
    }

    fn add(&self, other: &Self) -> Self {
        // a/b + c/d = (a·d + c·b)/(b·d)
        let num = super::poly_rde::poly_add(
            &poly_mul(&self.num, &other.den),
            &poly_mul(&other.num, &self.den),
        );
        let den = poly_mul(&self.den, &other.den);
        Self::new(num, den)
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(
            poly_mul(&self.num, &other.num),
            poly_mul(&self.den, &other.den),
        )
    }

    fn neg(&self) -> Self {
        Self::new(poly_scale(&self.num, &Rational::from(-1)), self.den.clone())
    }

    fn inv(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            // (num/den)⁻¹ = den/num; `new` re-canonicalizes the sign.
            Some(Self::new(self.den.clone(), self.num.clone()))
        }
    }

    /// `d/dx (num/den) = (num'·den − num·den') / den²`.
    fn derivative(&self) -> Self {
        let np = poly_deriv(&self.num);
        let dp = poly_deriv(&self.den);
        let numer = poly_sub(&poly_mul(&np, &self.den), &poly_mul(&self.num, &dp));
        let denom = poly_mul(&self.den, &self.den);
        Self::new(numer, denom)
    }
}

/// The rational-function field `ℚ(x)` — the coefficient field of an
/// [`AlgExtension`].  Its [`derivation`](CoeffField::derivation) is `d/dx`.
#[derive(Clone, Debug, Default)]
pub struct RationalFunctionField;

impl CoeffField for RationalFunctionField {
    type Elem = RatFn;

    fn zero(&self) -> RatFn {
        RatFn::int(0)
    }
    fn one(&self) -> RatFn {
        RatFn::int(1)
    }
    fn from_i64(&self, n: i64) -> RatFn {
        RatFn::int(n)
    }
    fn add(&self, a: &RatFn, b: &RatFn) -> RatFn {
        a.add(b)
    }
    fn sub(&self, a: &RatFn, b: &RatFn) -> RatFn {
        a.add(&b.neg())
    }
    fn mul(&self, a: &RatFn, b: &RatFn) -> RatFn {
        a.mul(b)
    }
    fn neg(&self, a: &RatFn) -> RatFn {
        a.neg()
    }
    fn inv(&self, a: &RatFn) -> Option<RatFn> {
        a.inv()
    }
    fn is_zero(&self, a: &RatFn) -> bool {
        a.is_zero()
    }
    fn eq(&self, a: &RatFn, b: &RatFn) -> bool {
        a == b
    }
    fn derivation(&self, a: &RatFn) -> RatFn {
        a.derivative()
    }
}

// ===========================================================================
// AlgExtension — ℚ(x)[y]/(q(x,y)) with a derivation
// ===========================================================================

/// An element of the extension: coefficients of `1, y, …, y^{d−1}`, each in
/// `ℚ(x)`.  (Generalizes the rank-2 `KPair`.)
pub type AlgElem = GPoly<RationalFunctionField>;

/// An algebraic function field extension `E = ℚ(x)[y]/(q(x, y))` of degree
/// `d = deg_y q`, with the derivation `D` extended from `d/dx` on `ℚ(x)` to `E`
/// via `D(y) = −q_x/q_y`.
#[derive(Clone, Debug)]
pub struct AlgExtension {
    quotient: Quotient<RationalFunctionField>,
    /// `D(y)`, the derivative of the generator, precomputed once.
    dy: AlgElem,
}

impl AlgExtension {
    /// Build `ℚ(x)[y]/(q)` from the minimal polynomial `q(x, y) = Σⱼ qⱼ(x) yʲ`,
    /// given as its coefficients `qⱼ(x) ∈ ℚ[x]` ascending in `y`.
    ///
    /// `q` should be squarefree and monic in `y`.  Panics if `q_y` (the formal
    /// `y`-derivative of `q`) is not invertible modulo `q` — i.e. if `q` is not
    /// separable, which never happens for a squarefree `q` in characteristic 0.
    pub fn new(q: &[QPoly]) -> Self {
        let modulus: AlgElem = q.iter().map(RatFn::from_poly).collect();
        let quotient = Quotient::new(RationalFunctionField, modulus);
        let dy = radical_dy(&quotient);
        Self { quotient, dy }
    }

    /// Build a simple radical extension `ℚ(x)[y]/(yⁿ − p(x))`, i.e. `y = p^{1/n}`.
    ///
    /// `n ≥ 2`; `p` must be a nonzero polynomial.  This is the common case the
    /// mixed-integration entry points produce (`∛x`, `x^{p/q}`, …).
    pub fn radical(n: usize, p: &QPoly) -> Self {
        debug_assert!(n >= 2, "radical extension needs degree ≥ 2");
        let mut q: Vec<QPoly> = vec![poly_zero(); n + 1];
        q[0] = poly_scale(p, &Rational::from(-1)); // −p
        q[n] = poly_one(); // yⁿ
        Self::new(&q)
    }

    /// The underlying generic quotient ring `ℚ(x)[y]/(q)`.
    pub fn quotient(&self) -> &Quotient<RationalFunctionField> {
        &self.quotient
    }

    /// Extension degree `d = deg_y q`.
    pub fn degree(&self) -> i64 {
        self.quotient.degree()
    }

    /// `D(y)`, the derivative of the algebraic generator.
    pub fn dy(&self) -> &AlgElem {
        &self.dy
    }

    /// The generator `y` as an element.
    pub fn generator(&self) -> AlgElem {
        vec![RatFn::int(0), RatFn::int(1)]
    }

    /// The constant element `r ∈ ℚ(x)`.
    pub fn constant(&self, r: RatFn) -> AlgElem {
        self.quotient.reduce(&[r])
    }

    /// Embed an integer.
    pub fn from_int(&self, n: i64) -> AlgElem {
        self.quotient.from_int(n)
    }

    /// Reduce an arbitrary `ℚ(x)`-polynomial in `y` mod `q`.
    pub fn reduce(&self, a: &[RatFn]) -> AlgElem {
        self.quotient.reduce(a)
    }

    /// `a + b`.
    pub fn add(&self, a: &[RatFn], b: &[RatFn]) -> AlgElem {
        self.quotient.add(a, b)
    }

    /// `a − b`.
    pub fn sub(&self, a: &[RatFn], b: &[RatFn]) -> AlgElem {
        self.quotient.sub(a, b)
    }

    /// `a · b`.
    pub fn mul(&self, a: &[RatFn], b: &[RatFn]) -> AlgElem {
        self.quotient.mul(a, b)
    }

    /// `−a`.
    pub fn neg(&self, a: &[RatFn]) -> AlgElem {
        self.quotient.neg(a)
    }

    /// `a⁻¹`, or `None` when `a` is a zero divisor / zero.
    pub fn inv(&self, a: &[RatFn]) -> Option<AlgElem> {
        self.quotient.inv(a)
    }

    /// `aⁿ` for any integer `n` (negative powers via [`inv`](AlgExtension::inv)).
    /// `None` only when `n < 0` and `a` is not invertible.
    pub fn pow(&self, a: &[RatFn], n: i64) -> Option<AlgElem> {
        if n == 0 {
            return Some(self.from_int(1));
        }
        if n < 0 {
            let inv = self.inv(a)?;
            return self.pow(&inv, -n);
        }
        let mut acc = self.from_int(1);
        for _ in 0..n {
            acc = self.mul(&acc, a);
        }
        Some(acc)
    }

    /// Is `a` equal to `b`?
    pub fn elem_eq(&self, a: &[RatFn], b: &[RatFn]) -> bool {
        self.quotient.elem_eq(a, b)
    }

    /// Is `a` zero?
    pub fn is_zero(&self, a: &[RatFn]) -> bool {
        self.quotient.elem_is_zero(a)
    }

    /// The derivation `D(a)` for `a = Σⱼ bⱼ(x) yʲ`:
    ///
    /// ```text
    ///   D(a) = Σⱼ bⱼ'(x) yʲ  +  (∂a/∂y) · D(y)        (reduced mod q)
    /// ```
    pub fn derivation(&self, a: &[RatFn]) -> AlgElem {
        quotient_derivation(&self.quotient, &self.dy, a)
    }
}

/// Formal derivative with respect to `y` of a polynomial-in-`y`
/// `Σⱼ cⱼ yʲ ↦ Σⱼ j cⱼ y^{j−1}`, with coefficients in `F` (the `cⱼ` treated as
/// constants).
pub(crate) fn dpoly_dy<F: CoeffField>(field: &F, p: &[F::Elem]) -> GPoly<F> {
    if p.len() <= 1 {
        return Vec::new();
    }
    p[1..]
        .iter()
        .enumerate()
        .map(|(i, c)| field.mul(&field.from_i64(i as i64 + 1), c))
        .collect()
}

/// The derivative `D(y)` of the generator of `Quotient<F>` with modulus
/// `q(x, y)`, forced by `q(x, y) = 0`: `D(y) = −q_x / q_y` where `q_x` applies
/// the coefficient field's derivation coefficient-wise and `q_y = ∂q/∂y`.
///
/// Generic over the coefficient field `F`, so it works both for `ℚ(x)`
/// ([`RationalFunctionField`]) and for a transcendental tower (the radicand
/// then involves the transcendental — the Risch MD case).
pub fn radical_dy<F: CoeffField>(quotient: &Quotient<F>) -> GPoly<F> {
    let field = quotient.field();
    let modulus = quotient.modulus();
    // q_x = ∂q/∂x  (coefficient-wise field derivation, holding y).
    let qx: GPoly<F> = modulus.iter().map(|c| field.derivation(c)).collect();
    // q_y = ∂q/∂y  (formal y-derivative).
    let qy = dpoly_dy(field, modulus);
    let qx_red = quotient.reduce(&qx);
    let qy_red = quotient.reduce(&qy);
    let qy_inv = quotient
        .inv(&qy_red)
        .expect("q is separable (squarefree, char 0): q_y is invertible mod q");
    quotient.mul(&quotient.neg(&qx_red), &qy_inv)
}

/// The extension derivation `D(a)` for `a = Σⱼ bⱼ yʲ` in `Quotient<F>` whose
/// generator has derivative `dy = D(y)`:
/// `D(a) = Σⱼ D(bⱼ) yʲ + (∂a/∂y)·D(y)` (reduced mod `q`).  Generic over `F`.
pub fn quotient_derivation<F: CoeffField>(
    quotient: &Quotient<F>,
    dy: &[F::Elem],
    a: &[F::Elem],
) -> GPoly<F> {
    let field = quotient.field();
    // Explicit (coefficient-field) part: differentiate each coefficient.
    let coeff_part: GPoly<F> = a.iter().map(|c| field.derivation(c)).collect();
    // Chain-rule part: (∂a/∂y) · D(y).
    let da_dy = dpoly_dy(field, a);
    let chain = quotient.mul(&da_dy, dy);
    quotient.add(&coeff_part, &chain)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rat(n: i64) -> Rational {
        Rational::from(n)
    }

    /// `1/(c·x^k)` as a `RatFn`.
    fn inv_monomial(c: i64, k: usize) -> RatFn {
        let mut den = vec![Rational::from(0); k + 1];
        den[k] = Rational::from(c);
        RatFn::new(vec![Rational::from(1)], den)
    }

    #[test]
    fn ratfn_canonicalizes() {
        // (2x)/(2) = x.
        let r = RatFn::new(vec![rat(0), rat(2)], vec![rat(2)]);
        assert_eq!(r, RatFn::from_poly(&vec![rat(0), rat(1)]));
        // 0/(x²) = 0/1.
        let z = RatFn::new(vec![], vec![rat(0), rat(0), rat(1)]);
        assert!(z.is_zero());
        assert_eq!(z.denom(), &poly_one());
    }

    #[test]
    fn ratfn_field_derivation_quotient_rule() {
        let f = RationalFunctionField;
        // d/dx (1/x) = −1/x².
        let one_over_x = inv_monomial(1, 1);
        let d = f.derivation(&one_over_x);
        assert_eq!(d, RatFn::new(vec![rat(-1)], vec![rat(0), rat(0), rat(1)]));
        // d/dx (x²) = 2x.
        let x2 = RatFn::from_poly(&vec![rat(0), rat(0), rat(1)]);
        assert_eq!(f.derivation(&x2), RatFn::from_poly(&vec![rat(0), rat(2)]));
    }

    #[test]
    fn derivation_of_cbrt_x() {
        // E = ℚ(x)(∛x) = ℚ(x)[y]/(y³ − x); modulus coeffs [−x, 0, 0, 1].
        // y³ = x ⇒ D(y) = 1/(3x) · y    (i.e. (1/3)x^{−2/3}).
        let e = AlgExtension::new(&[
            vec![rat(0), rat(-1)], // −x
            vec![rat(0)],
            vec![rat(0)],
            vec![rat(1)],
        ]);
        assert_eq!(e.degree(), 3);
        let y = e.generator();
        let dy = e.derivation(&y);
        // Expected: coefficient of y is 1/(3x), others zero.
        let expected = vec![RatFn::int(0), inv_monomial(3, 1), RatFn::int(0)];
        assert!(e.elem_eq(&dy, &expected), "D(∛x) = y/(3x); got {dy:?}");
        // And it equals the stored dy.
        assert!(e.elem_eq(e.dy(), &expected));
    }

    #[test]
    fn derivation_of_sqrt_x_matches_kpair() {
        // E = ℚ(x)(√x) = ℚ(x)[y]/(y² − x).  D(y) = 1/(2x)·y = (1/(2√x)).
        let e = AlgExtension::new(&[vec![rat(0), rat(-1)], vec![rat(0)], vec![rat(1)]]);
        let y = e.generator();
        let dy = e.derivation(&y);
        let expected = vec![RatFn::int(0), inv_monomial(2, 1)];
        assert!(e.elem_eq(&dy, &expected), "D(√x) = y/(2x); got {dy:?}");
    }

    #[test]
    fn derivation_consistency_y_cubed_is_x() {
        // D(y³) must reduce to D(x) = 1 (since y³ = x in the ring).
        let e = AlgExtension::new(&[
            vec![rat(0), rat(-1)], // −x
            vec![rat(0)],
            vec![rat(0)],
            vec![rat(1)],
        ]);
        let y = e.generator();
        let y3 = e.mul(&e.mul(&y, &y), &y); // = x  (reduced)
        let dy3 = e.derivation(&y3);
        assert!(e.elem_eq(&dy3, &e.from_int(1)), "D(y³)=D(x)=1; got {dy3:?}");
    }

    #[test]
    fn derivation_of_y_squared() {
        // y³ = x.  D(y²) = 2y·D(y) = 2y·(y/(3x)) = (2/(3x))·y².
        let e = AlgExtension::new(&[
            vec![rat(0), rat(-1)],
            vec![rat(0)],
            vec![rat(0)],
            vec![rat(1)],
        ]);
        let y = e.generator();
        let y2 = e.mul(&y, &y);
        let dy2 = e.derivation(&y2);
        // (2/(3x))·y²  →  coefficient of y² is 2/(3x).
        let two_over_3x = RatFn::new(vec![rat(2)], vec![rat(0), rat(3)]);
        let expected = vec![RatFn::int(0), RatFn::int(0), two_over_3x];
        assert!(e.elem_eq(&dy2, &expected), "D(y²); got {dy2:?}");
    }

    #[test]
    fn leibniz_rule_deterministic() {
        // D(α·β) = D(α)·β + α·D(β) for concrete α, β in ℚ(x)(∛x).
        let e = AlgExtension::new(&[
            vec![rat(0), rat(-1)],
            vec![rat(0)],
            vec![rat(0)],
            vec![rat(1)],
        ]);
        // α = x + y,  β = y² − 1/x.
        let alpha = vec![RatFn::from_poly(&vec![rat(0), rat(1)]), RatFn::int(1)];
        let beta = vec![inv_monomial(1, 1).neg(), RatFn::int(0), RatFn::int(1)];
        let lhs = e.derivation(&e.mul(&alpha, &beta));
        let rhs = e.add(
            &e.mul(&e.derivation(&alpha), &beta),
            &e.mul(&alpha, &e.derivation(&beta)),
        );
        assert!(e.elem_eq(&lhs, &rhs), "Leibniz: {lhs:?} vs {rhs:?}");
    }

    // -- proptest: ring axioms + Leibniz over ℚ(x)(∛x) --

    /// A small random `RatFn` with low-degree integer num/den.
    fn small_ratfn() -> impl Strategy<Value = RatFn> {
        let coeff = -3i64..=3;
        let nz = prop_oneof![1i64..=3, -3i64..=-1];
        (
            prop::collection::vec(coeff.clone(), 0..=2),
            prop::collection::vec(coeff, 0..=1),
            nz,
        )
            .prop_map(|(n, d, lead)| {
                let num: QPoly = n.into_iter().map(Rational::from).collect();
                let mut den: QPoly = d.into_iter().map(Rational::from).collect();
                den.push(Rational::from(lead)); // ensure nonzero denominator
                RatFn::new(num, den)
            })
    }

    fn small_elem() -> impl Strategy<Value = AlgElem> {
        // Degree < 3 element of ℚ(x)(∛x).
        prop::collection::vec(small_ratfn(), 0..=3)
    }

    fn cbrt_ext() -> AlgExtension {
        AlgExtension::new(&[
            vec![rat(0), rat(-1)],
            vec![rat(0)],
            vec![rat(0)],
            vec![rat(1)],
        ])
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_ring_axioms(a in small_elem(), b in small_elem(), c in small_elem()) {
            let e = cbrt_ext();
            // commutativity of + and ·
            prop_assert!(e.elem_eq(&e.add(&a, &b), &e.add(&b, &a)));
            prop_assert!(e.elem_eq(&e.mul(&a, &b), &e.mul(&b, &a)));
            // additive identity & inverse
            prop_assert!(e.elem_eq(&e.add(&a, &e.from_int(0)), &a));
            prop_assert!(e.is_zero(&e.add(&a, &e.neg(&a))));
            // multiplicative identity
            prop_assert!(e.elem_eq(&e.mul(&a, &e.from_int(1)), &a));
            // distributivity a·(b+c) = a·b + a·c
            let lhs = e.mul(&a, &e.add(&b, &c));
            let rhs = e.add(&e.mul(&a, &b), &e.mul(&a, &c));
            prop_assert!(e.elem_eq(&lhs, &rhs));
        }

        #[test]
        fn prop_leibniz(a in small_elem(), b in small_elem()) {
            let e = cbrt_ext();
            let lhs = e.derivation(&e.mul(&a, &b));
            let rhs = e.add(
                &e.mul(&e.derivation(&a), &b),
                &e.mul(&a, &e.derivation(&b)),
            );
            prop_assert!(e.elem_eq(&lhs, &rhs));
        }

        #[test]
        fn prop_derivation_is_additive(a in small_elem(), b in small_elem()) {
            let e = cbrt_ext();
            let lhs = e.derivation(&e.add(&a, &b));
            let rhs = e.add(&e.derivation(&a), &e.derivation(&b));
            prop_assert!(e.elem_eq(&lhs, &rhs));
        }
    }
}
