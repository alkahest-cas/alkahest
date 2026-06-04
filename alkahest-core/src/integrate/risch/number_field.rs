//! Algebraic number field `K = ℚ[t]/(q(t))` arithmetic, built on a **generic
//! polynomial-quotient core**.
//!
//! An element of `K` is represented as a [`QPoly`] of degree `< deg(q)` — a
//! ℚ-polynomial in the field generator `t`, reduced modulo the minimal
//! polynomial `q`.  A polynomial in a second variable `x` with coefficients in
//! `K` is a [`KPoly`] (`Vec<KElem>`, ascending degree in `x`).
//!
//! This module concentrates the quotient-ring primitives that were previously
//! private to [`super::rational_integrate`] (where they were introduced for the
//! Lazard–Rioboo–Trager degree-≥3 `RootSum` log argument, computing
//! `gcd_x(N − t·P', P)` in `ℚ(c)`).  They were first factored out here to back
//! the **ℚ(α)-coefficient RDE work** (Risch Gap E).
//!
//! ## Generic core (Risch M0)
//!
//! The quotient-ring arithmetic is *identical up to the coefficient ring* for
//! the constant case (coefficients in `ℚ`) and the function-field case
//! (coefficients in `ℚ(x)` rational functions): only the scalar type changes.
//! Accordingly the polynomial operations (`gpoly_*`, [`gext_gcd`],
//! [`gpoly_mod`], [`gmod_inverse`]) and the quotient ring [`Quotient`] are
//! generic over a [`CoeffField`] trait — the scalar field, optionally carrying a
//! `derivation`.  [`NumberField`] is then exactly the `CoeffField = `[`RationalField`]
//! (`ℚ`) instantiation, and the forthcoming `x`-dependent algebraic *function*
//! field (`alg_field.rs`, with a non-trivial derivation `D(y) = −q_x/q_y` over a
//! `ℚ(x)` base) is the other instantiation.  Both reuse this one implementation
//! of the polynomial-quotient plumbing rather than forking it.
//!
//! The three free functions [`poly_mod`], [`ext_gcd`], and [`mod_inverse`] are
//! the `ℚ`-coefficient specializations (the modulus need not be irreducible);
//! they underpin both [`NumberField`] and the Hermite-reduction step in
//! [`super::rational_integrate`].

use rug::Rational;

use super::poly_rde::{trim, QPoly};

// ===========================================================================
// CoeffField — the scalar field the quotient ring is built over
// ===========================================================================

/// A field of scalars over which polynomial / quotient-ring arithmetic is done.
///
/// Implementors so far: [`RationalField`] (`ℚ`, the constant field — its
/// [`derivation`](CoeffField::derivation) is identically zero).  The Risch M0
/// work adds a `ℚ(x)` rational-function instantiation whose derivation is
/// `d/dx`; that is what turns [`Quotient`] into an algebraic *function* field.
pub trait CoeffField {
    /// The scalar type.
    type Elem: Clone + std::fmt::Debug;

    /// The additive identity `0`.
    fn zero(&self) -> Self::Elem;
    /// The multiplicative identity `1`.
    fn one(&self) -> Self::Elem;
    /// Embed an integer.  (Takes `&self`: a coefficient field may need its own
    /// data to embed — e.g. a prime field reduces mod `p`.)
    #[allow(clippy::wrong_self_convention)]
    fn from_i64(&self, n: i64) -> Self::Elem;

    /// `a + b`.
    fn add(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `a − b`.
    fn sub(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `a · b`.
    fn mul(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    /// `−a`.
    fn neg(&self, a: &Self::Elem) -> Self::Elem;
    /// `a⁻¹`, or `None` if `a` is zero.  (A field: every nonzero element is a
    /// unit.)
    fn inv(&self, a: &Self::Elem) -> Option<Self::Elem>;

    /// Is `a` the zero element?
    fn is_zero(&self, a: &Self::Elem) -> bool;
    /// Are `a` and `b` equal?
    fn eq(&self, a: &Self::Elem, b: &Self::Elem) -> bool;

    /// The derivation `D` of the field.  Defaults to `0` — i.e. the field is
    /// treated as a field of *constants* (correct for `ℚ`).  An `x`-dependent
    /// coefficient field overrides this with `d/dx`.
    fn derivation(&self, a: &Self::Elem) -> Self::Elem {
        let _ = a;
        self.zero()
    }
}

/// The rational field `ℚ` — the constant coefficient field.
#[derive(Clone, Debug, Default)]
pub struct RationalField;

impl CoeffField for RationalField {
    type Elem = Rational;

    fn zero(&self) -> Rational {
        Rational::from(0)
    }
    fn one(&self) -> Rational {
        Rational::from(1)
    }
    fn from_i64(&self, n: i64) -> Rational {
        Rational::from(n)
    }
    fn add(&self, a: &Rational, b: &Rational) -> Rational {
        Rational::from(a + b)
    }
    fn sub(&self, a: &Rational, b: &Rational) -> Rational {
        Rational::from(a - b)
    }
    fn mul(&self, a: &Rational, b: &Rational) -> Rational {
        Rational::from(a * b)
    }
    fn neg(&self, a: &Rational) -> Rational {
        Rational::from(-a)
    }
    fn inv(&self, a: &Rational) -> Option<Rational> {
        if *a == 0 {
            None
        } else {
            Some(Rational::from(1) / a.clone())
        }
    }
    fn is_zero(&self, a: &Rational) -> bool {
        *a == 0
    }
    fn eq(&self, a: &Rational, b: &Rational) -> bool {
        a == b
    }
    // derivation defaults to 0 — ℚ is the constant field for d/dx.
}

// ===========================================================================
// Generic polynomial arithmetic over an arbitrary CoeffField
// ===========================================================================

/// A polynomial over `F` (ascending degree), as a coefficient vector.
pub type GPoly<F> = Vec<<F as CoeffField>::Elem>;

/// A polynomial in `x` whose coefficients live in a [`Quotient`]`<F>`
/// (ascending degree in `x`).
pub type KPolyG<F> = Vec<GPoly<F>>;

/// Drop trailing zero coefficients.
pub fn gtrim<F: CoeffField>(f: &F, mut p: GPoly<F>) -> GPoly<F> {
    while p.last().is_some_and(|c| f.is_zero(c)) {
        p.pop();
    }
    p
}

/// Degree (returns `-1` for the zero polynomial).
pub fn gdegree<F: CoeffField>(f: &F, p: &[F::Elem]) -> i64 {
    let mut d = p.len() as i64 - 1;
    while d >= 0 && f.is_zero(&p[d as usize]) {
        d -= 1;
    }
    d
}

/// The zero polynomial.
pub fn gpoly_zero<F: CoeffField>() -> GPoly<F> {
    Vec::new()
}

/// The constant polynomial `1`.
pub fn gpoly_one<F: CoeffField>(f: &F) -> GPoly<F> {
    vec![f.one()]
}

/// `a + b`.
pub fn gpoly_add<F: CoeffField>(f: &F, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
    let n = a.len().max(b.len());
    let mut r: GPoly<F> = (0..n).map(|_| f.zero()).collect();
    for (i, c) in a.iter().enumerate() {
        r[i] = f.add(&r[i], c);
    }
    for (i, c) in b.iter().enumerate() {
        r[i] = f.add(&r[i], c);
    }
    gtrim(f, r)
}

/// `a − b`.
pub fn gpoly_sub<F: CoeffField>(f: &F, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
    let n = a.len().max(b.len());
    let mut r: GPoly<F> = (0..n).map(|_| f.zero()).collect();
    for (i, c) in a.iter().enumerate() {
        r[i] = f.add(&r[i], c);
    }
    for (i, c) in b.iter().enumerate() {
        r[i] = f.sub(&r[i], c);
    }
    gtrim(f, r)
}

/// Scale `p` by the scalar `s`.
pub fn gpoly_scale<F: CoeffField>(f: &F, p: &[F::Elem], s: &F::Elem) -> GPoly<F> {
    if f.is_zero(s) || p.is_empty() {
        return Vec::new();
    }
    gtrim(f, p.iter().map(|c| f.mul(c, s)).collect())
}

/// `a · b`.
pub fn gpoly_mul<F: CoeffField>(f: &F, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r: GPoly<F> = (0..a.len() + b.len() - 1).map(|_| f.zero()).collect();
    for (i, ca) in a.iter().enumerate() {
        if f.is_zero(ca) {
            continue;
        }
        for (j, cb) in b.iter().enumerate() {
            let p = f.mul(ca, cb);
            r[i + j] = f.add(&r[i + j], &p);
        }
    }
    gtrim(f, r)
}

/// Long division: returns `(q, r)` with `a = q·b + r`, `deg r < deg b`.  `b`
/// must be nonzero; its leading coefficient must be invertible (always true in
/// a field).
pub fn gpoly_divrem<F: CoeffField>(f: &F, a: &[F::Elem], b: &[F::Elem]) -> (GPoly<F>, GPoly<F>) {
    let b = gtrim(f, b.to_vec());
    let bd = gdegree(f, &b);
    debug_assert!(bd >= 0, "gpoly_divrem: division by zero polynomial");
    let lc_inv = f
        .inv(&b[bd as usize])
        .expect("leading coefficient of a field element is invertible");

    let mut r = gtrim(f, a.to_vec());
    let ad = gdegree(f, &r);
    if ad < bd {
        return (Vec::new(), r);
    }
    let mut q: GPoly<F> = (0..(ad - bd + 1) as usize).map(|_| f.zero()).collect();
    loop {
        let rd = gdegree(f, &r);
        if rd < bd {
            break;
        }
        let shift = (rd - bd) as usize;
        let factor = f.mul(&r[rd as usize], &lc_inv);
        q[shift] = f.add(&q[shift], &factor);
        for (i, bc) in b.iter().enumerate() {
            let prod = f.mul(&factor, bc);
            r[shift + i] = f.sub(&r[shift + i], &prod);
        }
        r = gtrim(f, r);
        if r.is_empty() {
            break;
        }
    }
    (gtrim(f, q), r)
}

/// Extended GCD: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
pub fn gext_gcd<F: CoeffField>(
    f: &F,
    a: &[F::Elem],
    b: &[F::Elem],
) -> (GPoly<F>, GPoly<F>, GPoly<F>) {
    let (mut old_r, mut r) = (gtrim(f, a.to_vec()), gtrim(f, b.to_vec()));
    let (mut old_s, mut s) = (gpoly_one(f), gpoly_zero::<F>());
    let (mut old_t, mut t) = (gpoly_zero::<F>(), gpoly_one(f));
    while !r.is_empty() {
        let (q, rem) = gpoly_divrem(f, &old_r, &r);
        old_r = r;
        r = rem;
        let ns = gpoly_sub(f, &old_s, &gpoly_mul(f, &q, &s));
        old_s = s;
        s = ns;
        let nt = gpoly_sub(f, &old_t, &gpoly_mul(f, &q, &t));
        old_t = t;
        t = nt;
    }
    let dg = gdegree(f, &old_r);
    if dg < 0 {
        return (Vec::new(), old_s, old_t);
    }
    let inv = f
        .inv(&old_r[dg as usize])
        .expect("nonzero leading coefficient of a field element is invertible");
    (
        gpoly_scale(f, &old_r, &inv),
        gpoly_scale(f, &old_s, &inv),
        gpoly_scale(f, &old_t, &inv),
    )
}

/// Remainder of `a mod m`.
pub fn gpoly_mod<F: CoeffField>(f: &F, a: &[F::Elem], m: &[F::Elem]) -> GPoly<F> {
    gpoly_divrem(f, a, m).1
}

/// Inverse of `w` modulo `v` (requires `gcd(w, v) = 1`), else `None`.
pub fn gmod_inverse<F: CoeffField>(f: &F, w: &[F::Elem], v: &[F::Elem]) -> Option<GPoly<F>> {
    let (g, s, _t) = gext_gcd(f, w, v);
    if gdegree(f, &g) != 0 {
        return None; // not coprime
    }
    Some(gpoly_mod(f, &s, v))
}

/// Coefficient-wise equality after trimming.
pub fn gpoly_eq<F: CoeffField>(f: &F, a: &[F::Elem], b: &[F::Elem]) -> bool {
    let a = gtrim(f, a.to_vec());
    let b = gtrim(f, b.to_vec());
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f.eq(x, y))
}

// ---------------------------------------------------------------------------
// Base field: ℚ[x] modular arithmetic (the RationalField specialization)
// ---------------------------------------------------------------------------

/// Remainder of `a mod m` in `ℚ[x]`.
pub fn poly_mod(a: &QPoly, m: &QPoly) -> QPoly {
    gpoly_mod(&RationalField, a, m)
}

/// Extended GCD over `ℚ[x]`: returns `(g, s, t)` with `s·a + t·b = g`, `g` monic.
pub fn ext_gcd(a: &QPoly, b: &QPoly) -> (QPoly, QPoly, QPoly) {
    gext_gcd(&RationalField, a, b)
}

/// Inverse of `w` modulo `v` (requires `gcd(w, v) = 1`), else `None`.
pub fn mod_inverse(w: &QPoly, v: &QPoly) -> Option<QPoly> {
    gmod_inverse(&RationalField, w, v)
}

// ===========================================================================
// Generic quotient ring  F[t]/(modulus)  and  F[t]/(q) [x]  arithmetic
// ===========================================================================

/// The quotient ring `F[t]/(modulus)` over a coefficient field `F`, together
/// with arithmetic of polynomials in a second variable `x` over that quotient.
///
/// With `F = `[`RationalField`] this is an algebraic *number* field `ℚ[t]/(q)`
/// (see [`NumberField`]).  With an `x`-dependent `F = ℚ(x)` and a non-trivial
/// [`CoeffField::derivation`] it is an algebraic *function* field — the M0
/// substrate `alg_field.rs` builds on.
#[derive(Clone, Debug)]
pub struct Quotient<F: CoeffField> {
    field: F,
    modulus: GPoly<F>,
}

impl<F: CoeffField> Quotient<F> {
    /// Build `F[t]/(modulus)`.
    pub fn new(field: F, modulus: GPoly<F>) -> Self {
        let modulus = gtrim(&field, modulus);
        Self { field, modulus }
    }

    /// The coefficient field `F`.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// The defining (minimal) polynomial.
    pub fn modulus(&self) -> &GPoly<F> {
        &self.modulus
    }

    /// Extension degree `deg(modulus)`.
    pub fn degree(&self) -> i64 {
        gdegree(&self.field, &self.modulus)
    }

    /// The zero element of the quotient.
    pub fn elem_zero() -> GPoly<F> {
        Vec::new()
    }

    /// Is the quotient element zero?
    pub fn elem_is_zero(&self, a: &[F::Elem]) -> bool {
        gtrim(&self.field, a.to_vec()).is_empty()
    }

    /// Coefficient-wise equality of two quotient elements.
    pub fn elem_eq(&self, a: &[F::Elem], b: &[F::Elem]) -> bool {
        gpoly_eq(&self.field, a, b)
    }

    /// Reduce an arbitrary `F`-polynomial in `t` into canonical form.
    pub fn reduce(&self, a: &[F::Elem]) -> GPoly<F> {
        gpoly_mod(&self.field, a, &self.modulus)
    }

    /// `a + b`.
    pub fn add(&self, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
        self.reduce(&gpoly_add(&self.field, a, b))
    }

    /// `a − b`.
    pub fn sub(&self, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
        self.reduce(&gpoly_sub(&self.field, a, b))
    }

    /// `a · b`.
    pub fn mul(&self, a: &[F::Elem], b: &[F::Elem]) -> GPoly<F> {
        self.reduce(&gpoly_mul(&self.field, a, b))
    }

    /// `−a`.
    pub fn neg(&self, a: &[F::Elem]) -> GPoly<F> {
        let neg_one = self.field.neg(&self.field.one());
        self.reduce(&gpoly_scale(&self.field, a, &neg_one))
    }

    /// `a⁻¹`, or `None` when `a` is a zero divisor or zero.
    pub fn inv(&self, a: &[F::Elem]) -> Option<GPoly<F>> {
        gmod_inverse(&self.field, a, &self.modulus)
    }

    /// Embed an integer.
    pub fn from_int(&self, n: i64) -> GPoly<F> {
        self.reduce(&[self.field.from_i64(n)])
    }

    // -- polynomials in x over the quotient --

    /// Degree (in `x`) of a quotient-polynomial; `-1` for the zero polynomial.
    pub fn kdeg(&self, p: &[GPoly<F>]) -> i64 {
        let mut d = p.len() as i64 - 1;
        while d >= 0 && self.elem_is_zero(&p[d as usize]) {
            d -= 1;
        }
        d
    }

    /// Drop trailing zero quotient-coefficients.
    pub fn kpoly_trim(&self, mut p: Vec<GPoly<F>>) -> Vec<GPoly<F>> {
        while p.last().is_some_and(|c| self.elem_is_zero(c)) {
            p.pop();
        }
        p
    }

    /// Coefficient of `x^i`; the zero element for `i < 0` or out of range.
    pub fn kcoeff(&self, p: &[GPoly<F>], i: i64) -> GPoly<F> {
        if i < 0 {
            return Self::elem_zero();
        }
        p.get(i as usize).cloned().unwrap_or_else(Self::elem_zero)
    }

    /// Euclidean division of quotient-polynomials in `x`; `(quot, rem)` with
    /// `deg_x rem < deg_x b`.  `None` if `b` is zero or a leading coefficient is
    /// not invertible.
    pub fn kpoly_divrem(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Option<(KPolyG<F>, KPolyG<F>)> {
        let bd = self.kdeg(b);
        if bd < 0 {
            return None;
        }
        let lead_inv = self.inv(&b[bd as usize])?;
        let mut r = self.kpoly_trim(a.to_vec());
        let ad = self.kdeg(&r);
        if ad < bd {
            return Some((vec![], r));
        }
        let mut quot = vec![Self::elem_zero(); (ad - bd + 1) as usize];
        loop {
            let rd = self.kdeg(&r);
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
            r = self.kpoly_trim(r);
            if r.is_empty() {
                break;
            }
        }
        Some((self.kpoly_trim(quot), r))
    }

    /// Monic (in `x`) GCD of two quotient-polynomials.
    pub fn kpoly_gcd(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Option<Vec<GPoly<F>>> {
        let mut a = self.kpoly_trim(a.to_vec());
        let mut b = self.kpoly_trim(b.to_vec());
        while self.kdeg(&b) >= 0 {
            let (_, rem) = self.kpoly_divrem(&a, &b)?;
            a = b;
            b = rem;
        }
        let ad = self.kdeg(&a);
        if ad < 0 {
            return None;
        }
        let lead_inv = self.inv(&a[ad as usize])?;
        Some(a.iter().map(|c| self.mul(c, &lead_inv)).collect())
    }

    /// `a + b` of two quotient-polynomials in `x`.
    pub fn kpoly_add(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Vec<GPoly<F>> {
        let n = a.len().max(b.len());
        let mut r = vec![Self::elem_zero(); n];
        for (i, c) in a.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        for (i, c) in b.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        self.kpoly_trim(r)
    }

    /// `a − b` of two quotient-polynomials in `x`.
    pub fn kpoly_sub(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Vec<GPoly<F>> {
        let n = a.len().max(b.len());
        let mut r = vec![Self::elem_zero(); n];
        for (i, c) in a.iter().enumerate() {
            r[i] = self.add(&r[i], c);
        }
        for (i, c) in b.iter().enumerate() {
            r[i] = self.sub(&r[i], c);
        }
        self.kpoly_trim(r)
    }

    /// Scale a quotient-polynomial in `x` by a quotient element.
    pub fn kpoly_scale(&self, p: &[GPoly<F>], s: &[F::Elem]) -> Vec<GPoly<F>> {
        if self.elem_is_zero(s) {
            return Vec::new();
        }
        self.kpoly_trim(p.iter().map(|c| self.mul(c, s)).collect())
    }

    /// `a · b` of two quotient-polynomials in `x`.
    pub fn kpoly_mul(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Vec<GPoly<F>> {
        if self.kdeg(a) < 0 || self.kdeg(b) < 0 {
            return Vec::new();
        }
        let mut r = vec![Self::elem_zero(); a.len() + b.len() - 1];
        for (i, ca) in a.iter().enumerate() {
            if self.elem_is_zero(ca) {
                continue;
            }
            for (j, cb) in b.iter().enumerate() {
                let p = self.mul(ca, cb);
                r[i + j] = self.add(&r[i + j], &p);
            }
        }
        self.kpoly_trim(r)
    }

    /// `d/dx` of a quotient-polynomial in `x`.
    ///
    /// Note: this differentiates only with respect to the explicit `x`; it does
    /// **not** apply the coefficient field's own [`CoeffField::derivation`] to
    /// the coefficients (correct for the `ℚ`/constant case).  A function-field
    /// derivation that also acts on coefficients is layered on top in M0.
    pub fn kpoly_deriv(&self, p: &[GPoly<F>]) -> Vec<GPoly<F>> {
        if p.len() <= 1 {
            return Vec::new();
        }
        self.kpoly_trim(
            p[1..]
                .iter()
                .enumerate()
                .map(|(i, c)| self.mul(&self.from_int(i as i64 + 1), c))
                .collect(),
        )
    }

    /// `∫ dx` of a quotient-polynomial in `x` (constant of integration 0).
    pub fn kpoly_integrate(&self, p: &[GPoly<F>]) -> Vec<GPoly<F>> {
        let p = self.kpoly_trim(p.to_vec());
        if p.is_empty() {
            return Vec::new();
        }
        let mut r = vec![Self::elem_zero()];
        for (i, c) in p.iter().enumerate() {
            let inv = self
                .inv(&self.from_int(i as i64 + 1))
                .expect("nonzero integer is invertible in a field");
            r.push(self.mul(c, &inv));
        }
        self.kpoly_trim(r)
    }

    /// `p^n` of a quotient-polynomial in `x` (`n ≥ 0`; `p^0 = 1`).
    pub fn kpoly_pow(&self, p: &[GPoly<F>], n: u32) -> Vec<GPoly<F>> {
        let mut acc = vec![self.from_int(1)];
        for _ in 0..n {
            acc = self.kpoly_mul(&acc, p);
        }
        acc
    }

    /// Make a quotient-polynomial monic in `x`.  The zero polynomial is returned
    /// unchanged; `None` if the leading coefficient is not invertible.
    pub fn kpoly_monic(&self, p: &[GPoly<F>]) -> Option<Vec<GPoly<F>>> {
        let d = self.kdeg(p);
        if d < 0 {
            return Some(Vec::new());
        }
        let inv = self.inv(&p[d as usize])?;
        Some(self.kpoly_trim(p.iter().map(|c| self.mul(c, &inv)).collect()))
    }

    /// Exact division `a / b`; `None` if `b` does not divide `a` evenly.
    pub fn kpoly_div_exact(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> Option<Vec<GPoly<F>>> {
        let (q, r) = self.kpoly_divrem(a, b)?;
        if self.kdeg(&r) >= 0 {
            return None;
        }
        Some(q)
    }

    /// Equality of two quotient-polynomials in `x` (coefficient-wise after
    /// trimming).
    pub fn kpoly_eq(&self, a: &[GPoly<F>], b: &[GPoly<F>]) -> bool {
        let a = self.kpoly_trim(a.to_vec());
        let b = self.kpoly_trim(b.to_vec());
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| self.elem_eq(x, y))
    }
}

// ===========================================================================
// NumberField = the ℚ instantiation  Quotient<RationalField>
// ===========================================================================

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
///
/// This is a thin wrapper over [`Quotient`]`<`[`RationalField`]`>`: the algebraic
/// number field is exactly the constant-coefficient (`ℚ`) instantiation of the
/// generic polynomial-quotient core.  The number-theoretic surface
/// ([`norm`](NumberField::norm), [`trace`](NumberField::trace),
/// [`field_degree`](NumberField::field_degree)) lives here because it is
/// specific to the constant field.
#[derive(Clone, Debug)]
pub struct NumberField {
    inner: Quotient<RationalField>,
}

impl NumberField {
    /// Build `K = ℚ[t]/(modulus)`.
    pub fn new(modulus: QPoly) -> Self {
        Self {
            inner: Quotient::new(RationalField, modulus),
        }
    }

    /// The underlying generic quotient ring.
    pub fn quotient(&self) -> &Quotient<RationalField> {
        &self.inner
    }

    /// The defining (minimal) polynomial `q(t)`.
    pub fn modulus(&self) -> &QPoly {
        self.inner.modulus()
    }

    /// Field degree `[K : ℚ] = deg q`.
    pub fn degree(&self) -> i64 {
        self.inner.degree()
    }

    /// Field degree `[K : ℚ]` (alias of [`degree`](NumberField::degree); part of
    /// the §7.3 algebraic-number-field surface).
    pub fn field_degree(&self) -> i64 {
        self.inner.degree()
    }

    /// Reduce an arbitrary ℚ-polynomial in `t` into canonical `K`-element form.
    pub fn reduce(&self, a: &KElem) -> KElem {
        self.inner.reduce(a)
    }

    /// `a + b` in `K`.
    pub fn add(&self, a: &KElem, b: &KElem) -> KElem {
        self.inner.add(a, b)
    }

    /// `a − b` in `K`.
    pub fn sub(&self, a: &KElem, b: &KElem) -> KElem {
        self.inner.sub(a, b)
    }

    /// `a · b` in `K`.
    pub fn mul(&self, a: &KElem, b: &KElem) -> KElem {
        self.inner.mul(a, b)
    }

    /// `−a` in `K`.
    pub fn neg(&self, a: &KElem) -> KElem {
        self.inner.neg(a)
    }

    /// Multiplicative inverse `a⁻¹` in `K`, or `None` when `a` is a zero divisor
    /// (e.g. shares a factor with a non-irreducible modulus) or zero.
    pub fn inv(&self, a: &KElem) -> Option<KElem> {
        self.inner.inv(a)
    }

    /// Is the `K`-element zero?
    pub fn is_zero(a: &KElem) -> bool {
        trim(a.clone()).is_empty()
    }

    // -- polynomials in x over K (delegated to the generic quotient) --

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
        self.inner.kpoly_divrem(a, b)
    }

    /// Monic (in `x`) GCD of two `K`-polynomials.  `None` if both are zero or a
    /// leading coefficient is not invertible in `K`.
    pub fn kpoly_gcd(&self, a: &[KElem], b: &[KElem]) -> Option<KPoly> {
        self.inner.kpoly_gcd(a, b)
    }

    /// The zero `K`-element.
    pub fn k_zero() -> KElem {
        Vec::new()
    }

    /// Embed an integer into `K`.
    pub fn from_int(&self, n: i64) -> KElem {
        self.inner.from_int(n)
    }

    /// Embed a rational into `K`.
    pub fn from_rational(&self, r: &Rational) -> KElem {
        self.inner.reduce(std::slice::from_ref(r))
    }

    /// `a + b` of two `K`-polynomials in `x`.
    pub fn kpoly_add(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        self.inner.kpoly_add(a, b)
    }

    /// `a − b` of two `K`-polynomials in `x`.
    pub fn kpoly_sub(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        self.inner.kpoly_sub(a, b)
    }

    /// Scale a `K`-polynomial in `x` by a `K`-element.
    pub fn kpoly_scale(&self, p: &[KElem], s: &KElem) -> KPoly {
        self.inner.kpoly_scale(p, s)
    }

    /// `a · b` of two `K`-polynomials in `x`.
    pub fn kpoly_mul(&self, a: &[KElem], b: &[KElem]) -> KPoly {
        self.inner.kpoly_mul(a, b)
    }

    /// `d/dx` of a `K`-polynomial in `x`.
    pub fn kpoly_deriv(&self, p: &[KElem]) -> KPoly {
        self.inner.kpoly_deriv(p)
    }

    /// `∫ dx` of a `K`-polynomial in `x` (constant of integration 0).
    pub fn kpoly_integrate(&self, p: &[KElem]) -> KPoly {
        self.inner.kpoly_integrate(p)
    }

    /// `p^n` of a `K`-polynomial in `x` (`n ≥ 0`; `p^0 = 1`).
    pub fn kpoly_pow(&self, p: &[KElem], n: u32) -> KPoly {
        self.inner.kpoly_pow(p, n)
    }

    /// Make a `K`-polynomial monic in `x` (divide by its leading coefficient).
    /// The zero polynomial is returned unchanged; `None` if the leading
    /// coefficient is not invertible in `K`.
    pub fn kpoly_monic(&self, p: &[KElem]) -> Option<KPoly> {
        self.inner.kpoly_monic(p)
    }

    /// Exact division `a / b` of `K`-polynomials in `x`; `None` if `b` does not
    /// divide `a` evenly (or `b` is zero / has a non-invertible leading term).
    pub fn kpoly_div_exact(&self, a: &[KElem], b: &[KElem]) -> Option<KPoly> {
        self.inner.kpoly_div_exact(a, b)
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

    // -- number-theoretic surface (mathematical-coverage.md §7.3 slice) --

    /// The multiplication-by-`a` matrix `M` of the ℚ-linear endomorphism `b ↦ a·b`
    /// of `K`, in the power basis `1, t, …, t^{d−1}`.  `M[i][j]` is the
    /// coefficient of `t^i` in `a·t^j` reduced mod `q`.
    fn mult_matrix(&self, a: &KElem) -> Vec<Vec<Rational>> {
        let d = self.degree();
        if d < 1 {
            // Degenerate: K is (a quotient of) ℚ itself.
            let c = self.reduce(a);
            let v = c.first().cloned().unwrap_or_else(|| Rational::from(0));
            return vec![vec![v]];
        }
        let d = d as usize;
        let a_red = self.reduce(a);
        let mut m = vec![vec![Rational::from(0); d]; d];
        for j in 0..d {
            // a · t^j, reduced mod q, gives column j in the power basis.
            let mut mono = vec![Rational::from(0); j + 1];
            mono[j] = Rational::from(1);
            let col = self.mul(&a_red, &mono);
            for (i, row) in m.iter_mut().enumerate() {
                row[j] = col.get(i).cloned().unwrap_or_else(|| Rational::from(0));
            }
        }
        m
    }

    /// Field norm `N_{K/ℚ}(a)` — the determinant of multiplication by `a`.
    pub fn norm(&self, a: &KElem) -> Rational {
        rat_det(self.mult_matrix(a))
    }

    /// Field trace `Tr_{K/ℚ}(a)` — the trace of multiplication by `a`.
    pub fn trace(&self, a: &KElem) -> Rational {
        let m = self.mult_matrix(a);
        m.iter()
            .enumerate()
            .fold(Rational::from(0), |s, (i, row)| s + row[i].clone())
    }
}

/// Determinant of a square rational matrix via Gaussian elimination.
fn rat_det(mut m: Vec<Vec<Rational>>) -> Rational {
    let n = m.len();
    let mut det = Rational::from(1);
    for col in 0..n {
        let Some(piv) = (col..n).find(|&r| m[r][col] != 0) else {
            return Rational::from(0);
        };
        if piv != col {
            m.swap(piv, col);
            det = -det;
        }
        let pivot = m[col][col].clone();
        det *= pivot.clone();
        let pivot_row: Vec<Rational> = m[col][col..n].to_vec();
        for row in m.iter_mut().skip(col + 1) {
            if row[col] != 0 {
                let factor = row[col].clone() / pivot.clone();
                for (dst, src) in row[col..n].iter_mut().zip(pivot_row.iter()) {
                    *dst -= factor.clone() * src.clone();
                }
            }
        }
    }
    det
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::risch::poly_rde::{poly_mul, poly_zero};

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

    // -- M0 additions --

    #[test]
    fn rational_field_derivation_is_zero() {
        // ℚ is the constant field: D(a) = 0 for all a.
        let f = RationalField;
        assert!(f.is_zero(&f.derivation(&rat(5))));
        assert!(f.is_zero(&f.derivation(&frac(3, 7))));
    }

    #[test]
    fn norm_and_trace_over_sqrt2() {
        let k = q_sqrt2();
        assert_eq!(k.field_degree(), 2);
        let t = vec![rat(0), rat(1)]; // √2
                                      // N(√2) = −2, Tr(√2) = 0.
        assert_eq!(k.norm(&t), rat(-2));
        assert_eq!(k.trace(&t), rat(0));
        // N(1+√2) = (1+√2)(1−√2) = −1, Tr(1+√2) = 2.
        let one_plus = vec![rat(1), rat(1)];
        assert_eq!(k.norm(&one_plus), rat(-1));
        assert_eq!(k.trace(&one_plus), rat(2));
        // N(3) = 3² = 9 for a degree-2 field, Tr(3) = 6.
        let three = vec![rat(3)];
        assert_eq!(k.norm(&three), rat(9));
        assert_eq!(k.trace(&three), rat(6));
    }

    #[test]
    fn norm_over_cube_root_field() {
        // K = ℚ(∛2) = ℚ[t]/(t³−2); N(∛2) = 2 (= constant term, up to sign for
        // t³−2 the norm of the generator is 2).
        let k = NumberField::new(vec![rat(-2), rat(0), rat(0), rat(1)]);
        assert_eq!(k.field_degree(), 3);
        let cbrt2 = vec![rat(0), rat(1)];
        assert_eq!(k.norm(&cbrt2), rat(2));
        assert_eq!(k.trace(&cbrt2), rat(0));
    }

    /// A second `CoeffField` — the prime field `F_p` — to exercise the generic
    /// polynomial core over a non-ℚ field (the whole point of M0's
    /// generalization).
    #[derive(Clone, Debug)]
    struct Fp {
        p: i64,
    }

    fn modp(a: i64, p: i64) -> i64 {
        ((a % p) + p) % p
    }
    fn modpow(mut b: i64, mut e: i64, p: i64) -> i64 {
        let mut acc = 1i64;
        b = modp(b, p);
        while e > 0 {
            if e & 1 == 1 {
                acc = modp(acc * b, p);
            }
            b = modp(b * b, p);
            e >>= 1;
        }
        acc
    }

    impl CoeffField for Fp {
        type Elem = i64;
        fn zero(&self) -> i64 {
            0
        }
        fn one(&self) -> i64 {
            1 % self.p
        }
        fn from_i64(&self, n: i64) -> i64 {
            modp(n, self.p)
        }
        fn add(&self, a: &i64, b: &i64) -> i64 {
            modp(a + b, self.p)
        }
        fn sub(&self, a: &i64, b: &i64) -> i64 {
            modp(a - b, self.p)
        }
        fn mul(&self, a: &i64, b: &i64) -> i64 {
            modp(a * b, self.p)
        }
        fn neg(&self, a: &i64) -> i64 {
            modp(-a, self.p)
        }
        fn inv(&self, a: &i64) -> Option<i64> {
            if modp(*a, self.p) == 0 {
                None
            } else {
                Some(modpow(*a, self.p - 2, self.p)) // Fermat (p prime)
            }
        }
        fn is_zero(&self, a: &i64) -> bool {
            modp(*a, self.p) == 0
        }
        fn eq(&self, a: &i64, b: &i64) -> bool {
            modp(a - b, self.p) == 0
        }
    }

    #[test]
    fn generic_gcd_over_f5() {
        // Over F_5: gcd(x²−1, x−1) = x−1, returned monic.
        let f = Fp { p: 5 };
        let a = vec![f.from_i64(-1), f.from_i64(0), f.from_i64(1)]; // x²−1
        let b = vec![f.from_i64(-1), f.from_i64(1)]; // x−1
        let (g, _s, _t) = gext_gcd(&f, &a, &b);
        assert_eq!(gdegree(&f, &g), 1);
        // Monic x−1 over F_5 = [4, 1].
        assert_eq!(g, vec![4, 1]);
        // Bézout sanity: g | a and g | b.
        assert!(gpoly_mod(&f, &a, &g).is_empty() || gdegree(&f, &gpoly_mod(&f, &a, &g)) < 0);
        assert!(gdegree(&f, &gpoly_mod(&f, &b, &g)) < 0);
    }

    #[test]
    fn generic_quotient_over_f7() {
        // F_7[t]/(t²+1): t is a square root of −1 = 6.  t·t = −1 = 6.
        let f = Fp { p: 7 };
        let q = Quotient::new(f.clone(), vec![f.from_i64(1), f.from_i64(0), f.from_i64(1)]);
        let t = vec![0i64, 1];
        let tt = q.mul(&t, &t);
        assert!(q.elem_eq(&tt, &[6])); // t² = −1 = 6
                                       // Inverse of t: t·t⁻¹ = 1.  t⁻¹ = −t = 6t.
        let inv = q.inv(&t).expect("t invertible mod t²+1");
        assert!(q.elem_eq(&q.mul(&t, &inv), &[1]));
    }

    #[test]
    fn numberfield_matches_generic_quotient() {
        // NumberField delegates to Quotient<RationalField>; cross-check directly.
        let modulus = vec![rat(-2), rat(0), rat(1)]; // t²−2
        let nf = NumberField::new(modulus.clone());
        let q = Quotient::new(RationalField, modulus);
        let a = vec![rat(1), rat(1)]; // 1+√2
        assert_eq!(nf.mul(&a, &a), q.mul(&a, &a));
        assert_eq!(nf.inv(&a), q.inv(&a));
    }
}
