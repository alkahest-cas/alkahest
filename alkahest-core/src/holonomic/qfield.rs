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

/// `a + b`.
///
/// Both operands are assumed to be in `RatFunc`'s canonical form (coprime
/// `num`/`den`, monic `den`) — every constructor in this module produces it.
/// Given that, the only cancellation possible is against `gcd(a.den, b.den)`,
/// which is what makes the two small gcds below enough: the generic
/// `RatFunc + RatFunc` instead reduces the full cross-multiplied product, whose
/// degree is the *sum* of the two denominators'.
pub fn rn_add(a: &Rn, b: &Rn) -> Rn {
    if rn_is_zero(a) {
        return b.clone();
    }
    if rn_is_zero(b) {
        return a.clone();
    }
    let g = if a.den.degree() <= 0 || b.den.degree() <= 0 {
        RatUniPoly::one()
    } else {
        q_gcd(&a.den, &b.den)
    };
    if g.degree() <= 0 {
        // Coprime denominators: the sum is already in lowest terms.
        let num = &(&a.num * &b.den) + &(&b.num * &a.den);
        let den = &a.den * &b.den;
        return rn_from_coprime(num, den);
    }
    let (Some(b1), Some(d1)) = (q_exact_div(&a.den, &g), q_exact_div(&b.den, &g)) else {
        return a.clone() + b.clone();
    };
    // `num/den = (a.num·d1 + b.num·b1) / (a.den·d1)` over the lcm of the two
    // denominators; whatever is left to cancel must divide `g`.
    let num = &(&a.num * &d1) + &(&b.num * &b1);
    if num.is_zero() {
        return rn_zero();
    }
    let den = &a.den * &d1;
    let h = q_gcd(&num, &g);
    if h.degree() > 0 {
        if let (Some(nn), Some(dd)) = (q_exact_div(&num, &h), q_exact_div(&den, &h)) {
            return rn_from_coprime(nn, dd);
        }
    }
    rn_from_coprime(num, den)
}

pub fn rn_neg(a: &Rn) -> Rn {
    -a.clone()
}

pub fn rn_sub(a: &Rn, b: &Rn) -> Rn {
    rn_add(a, &rn_neg(b))
}

/// `a · b`.
///
/// Same canonical-form assumption as [`rn_add`]: with both operands reduced,
/// `gcd(a.num·b.num, a.den·b.den) = gcd(a.num, b.den)·gcd(b.num, a.den)`, so
/// cancelling crosswise *before* multiplying is both cheaper and complete.
pub fn rn_mul(a: &Rn, b: &Rn) -> Rn {
    if rn_is_zero(a) || rn_is_zero(b) {
        return rn_zero();
    }
    let (an, bd) = cross_cancel(&a.num, &b.den);
    let (bn, ad) = cross_cancel(&b.num, &a.den);
    rn_from_coprime(&an * &bn, &ad * &bd)
}

pub fn rn_inv(a: &Rn) -> Option<Rn> {
    if rn_is_zero(a) {
        return None;
    }
    // `a` is reduced, so the reciprocal is too — only the monic normalisation
    // of the new denominator is left to do.
    Some(rn_from_coprime(a.den.clone(), a.num.clone()))
}

/// Divide `gcd(x, y)` out of both, leaving them untouched when it is a unit.
fn cross_cancel(x: &RatUniPoly, y: &RatUniPoly) -> (RatUniPoly, RatUniPoly) {
    if x.degree() <= 0 || y.degree() <= 0 {
        return (x.clone(), y.clone());
    }
    let g = q_gcd(x, y);
    if g.degree() <= 0 {
        return (x.clone(), y.clone());
    }
    match (q_exact_div(x, &g), q_exact_div(y, &g)) {
        (Some(xx), Some(yy)) => (xx, yy),
        _ => (x.clone(), y.clone()),
    }
}

/// Build an `Rn` from an already-coprime pair, normalizing `den` to monic.
fn rn_from_coprime(num: RatUniPoly, den: RatUniPoly) -> Rn {
    let num = num.trim();
    if num.is_zero() {
        return rn_zero();
    }
    let den = den.trim();
    let lc = den.leading_coeff();
    if lc == 1 || den.is_zero() {
        return RatFunc { num, den };
    }
    let inv = Rational::from(1) / lc;
    RatFunc {
        num: scale_ratuni(&num, &inv),
        den: scale_ratuni(&den, &inv),
    }
}

fn scale_ratuni(p: &RatUniPoly, z: &Rational) -> RatUniPoly {
    if p.is_zero() || *z == 1 {
        return p.clone();
    }
    RatUniPoly {
        coeffs: p.coeffs.iter().map(|c| c.clone() * z.clone()).collect(),
    }
    .trim()
}

fn q_exact_div(a: &RatUniPoly, b: &RatUniPoly) -> Option<RatUniPoly> {
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

/// Monic gcd in `Q[n]`, computed by the `Z[n]` subresultant PRS.
///
/// `RatUniPoly::gcd` is the naive Euclidean sequence over `Q`, which swells the
/// rational coefficients of every remainder; clearing denominators once and
/// running the subresultant PRS in `Z[n]` keeps the whole computation in
/// integers. The result is the same monic polynomial.
fn q_gcd(a: &RatUniPoly, b: &RatUniPoly) -> RatUniPoly {
    if a.is_zero() {
        return make_monic(b.clone());
    }
    if b.is_zero() {
        return make_monic(a.clone());
    }
    if a.degree() <= 0 || b.degree() <= 0 {
        return RatUniPoly::one();
    }
    let g = IPoly::gcd(&ratuni_to_zpoly(a), &ratuni_to_zpoly(b));
    make_monic(ipoly_to_ratuni(&g))
}

fn make_monic(p: RatUniPoly) -> RatUniPoly {
    let p = p.trim();
    if p.is_zero() {
        return p;
    }
    let lc = p.leading_coeff();
    if lc == 1 {
        return p;
    }
    scale_ratuni(&p, &(Rational::from(1) / lc))
}

/// Clear the rational denominators of a `Q[n]` polynomial, giving a `Z[n]` one
/// that is a rational multiple of it (enough for gcd purposes).
fn ratuni_to_zpoly(p: &RatUniPoly) -> IPoly {
    let mut l = Integer::from(1);
    for c in &p.coeffs {
        l = l.lcm(&c.clone().denom().clone());
    }
    let scaled = scale_ratuni(p, &Rational::from(l));
    IPoly {
        c: scaled
            .coeffs
            .iter()
            .map(|c| c.numer().clone())
            .collect::<Vec<_>>(),
    }
    .trim()
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
// Z[n] and Z[n][k] — the integral-domain view that gcd actually runs in
// ---------------------------------------------------------------------------
//
// `Q(n)[k]` is a Euclidean domain, so a gcd *can* be computed by the textbook
// remainder sequence over the field `Q(n)` — and that is what this module used
// to do. It is also the textbook example of intermediate expression swell: each
// `div_rem` step divides by the leading coefficient, so every remainder
// coefficient is a fresh quotient of rational functions in `n` whose numerator
// and denominator degrees add, and nothing ever removes the common content that
// makes them cancel again. On Franel (`Σ_k C(n,k)³`) one such gcd ran for
// eleven seconds.
//
// The classical fix is to leave the field and work in the UFD `Z[n][k]`, where
// the coefficients stay polynomials and the swell is controlled by *dividing
// out the subresultant* at every step (Brown's subresultant PRS: Collins 1967,
// Brown 1971; Knuth, TAOCP vol. 2 § 4.6.1, algorithm C). The gcd over `Q(n)[k]`
// is the gcd over `Z[n][k]` up to a unit of `Q(n)`, which the monic
// normalisation at the end absorbs, so this is the same mathematical object,
// only reachable in milliseconds instead of seconds.

/// Dense univariate polynomial over `Z` in the outer variable `n`, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
struct IPoly {
    c: Vec<Integer>,
}

impl IPoly {
    fn zero() -> Self {
        IPoly { c: vec![] }
    }

    fn one() -> Self {
        IPoly {
            c: vec![Integer::from(1)],
        }
    }

    fn from_int(i: Integer) -> Self {
        if i == 0 {
            IPoly::zero()
        } else {
            IPoly { c: vec![i] }
        }
    }

    fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    fn is_one(&self) -> bool {
        self.c.len() == 1 && self.c[0] == 1
    }

    /// Degree, or `-1` for the zero polynomial.
    fn degree(&self) -> i32 {
        self.c.len() as i32 - 1
    }

    fn trim(mut self) -> Self {
        while self.c.last().map(|v| *v == 0).unwrap_or(false) {
            self.c.pop();
        }
        self
    }

    fn lc(&self) -> Integer {
        self.c.last().cloned().unwrap_or_else(|| Integer::from(0))
    }

    fn sub(&self, other: &Self) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.c.get(i).cloned().unwrap_or_else(|| Integer::from(0));
            let b = other.c.get(i).cloned().unwrap_or_else(|| Integer::from(0));
            out.push(a - b);
        }
        IPoly { c: out }.trim()
    }

    fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return IPoly::zero();
        }
        let mut out = vec![Integer::from(0); self.c.len() + other.c.len() - 1];
        for (i, a) in self.c.iter().enumerate() {
            if *a == 0 {
                continue;
            }
            for (j, b) in other.c.iter().enumerate() {
                if *b == 0 {
                    continue;
                }
                out[i + j] += Integer::from(a * b);
            }
        }
        IPoly { c: out }.trim()
    }

    fn scale_int(&self, z: &Integer) -> Self {
        if *z == 0 {
            return IPoly::zero();
        }
        IPoly {
            c: self.c.iter().map(|v| Integer::from(v * z)).collect(),
        }
        .trim()
    }

    fn pow_usize(&self, e: usize) -> Self {
        let mut acc = IPoly::one();
        for _ in 0..e {
            acc = acc.mul(self);
        }
        acc
    }

    /// `x^shift · self`.
    fn shift_pow(&self, shift: usize) -> Self {
        if self.is_zero() || shift == 0 {
            return self.clone();
        }
        let mut c = vec![Integer::from(0); shift];
        c.extend(self.c.iter().cloned());
        IPoly { c }
    }

    /// Non-negative gcd of the coefficients (`0` for the zero polynomial).
    fn content(&self) -> Integer {
        let mut g = Integer::from(0);
        for v in &self.c {
            g = g.gcd(v);
            if g == 1 {
                break;
            }
        }
        g.abs()
    }

    fn div_int_exact(&self, z: &Integer) -> Option<Self> {
        if *z == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(self.c.len());
        for v in &self.c {
            if !v.is_divisible(z) {
                return None;
            }
            out.push(Integer::from(v / z));
        }
        Some(IPoly { c: out }.trim())
    }

    /// Primitive part, normalized to a positive leading coefficient.
    fn primitive_part(&self) -> Self {
        if self.is_zero() {
            return IPoly::zero();
        }
        let mut cont = self.content();
        if self.lc() < 0 {
            cont = -cont;
        }
        self.div_int_exact(&cont).unwrap_or_else(|| self.clone())
    }

    /// Exact division in `Z[n]`; `None` when the quotient is not in `Z[n]`.
    fn exact_div(&self, d: &Self) -> Option<Self> {
        if d.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(IPoly::zero());
        }
        let dd = d.degree();
        if self.degree() < dd {
            return None;
        }
        let ld = d.lc();
        let mut rem = self.c.clone();
        let mut q = vec![Integer::from(0); (self.degree() - dd + 1) as usize];
        let mut i = self.degree();
        while i >= dd {
            let lr = rem[i as usize].clone();
            if lr != 0 {
                if !lr.is_divisible(&ld) {
                    return None;
                }
                let t = Integer::from(&lr / &ld);
                let shift = (i - dd) as usize;
                q[shift] = t.clone();
                for (j, dc) in d.c.iter().enumerate() {
                    rem[shift + j] -= Integer::from(&t * dc);
                }
            }
            i -= 1;
        }
        if rem[..dd as usize].iter().any(|v| *v != 0) {
            return None;
        }
        Some(IPoly { c: q }.trim())
    }

    /// Pseudo-remainder: the remainder of `lc(b)^(deg a − deg b + 1)·a` by `b`.
    fn pseudo_rem(a: &Self, b: &Self) -> Self {
        let db = b.degree();
        let mut r = a.clone().trim();
        if r.degree() < db || r.is_zero() {
            return r;
        }
        let lb = b.lc();
        let mut e = (r.degree() - db + 1) as usize;
        while !r.is_zero() && r.degree() >= db {
            let shift = (r.degree() - db) as usize;
            let lr = r.lc();
            r = r.scale_int(&lb).sub(&b.shift_pow(shift).scale_int(&lr));
            e -= 1;
        }
        for _ in 0..e {
            r = r.scale_int(&lb);
        }
        r
    }

    /// gcd in `Z[n]` by subresultant PRS, normalized to a positive leading
    /// coefficient (`0` only when both inputs are `0`).
    fn gcd(a: &Self, b: &Self) -> Self {
        let mut x = a.clone().trim();
        let mut y = b.clone().trim();
        if x.is_zero() {
            return y.primitive_part().scale_int(&y.content());
        }
        if y.is_zero() {
            return x.primitive_part().scale_int(&x.content());
        }
        if x.degree() < y.degree() {
            std::mem::swap(&mut x, &mut y);
        }
        let cx = x.content();
        let cy = y.content();
        let d = Integer::from(cx.gcd_ref(&cy));
        x = x.div_int_exact(&cx).unwrap_or(x);
        y = y.div_int_exact(&cy).unwrap_or(y);
        if y.degree() == 0 {
            return IPoly::from_int(d);
        }
        let mut g = Integer::from(1);
        let mut h = Integer::from(1);
        loop {
            let delta = (x.degree() - y.degree()) as usize;
            let r = IPoly::pseudo_rem(&x, &y);
            if r.is_zero() {
                break;
            }
            if r.degree() == 0 {
                y = IPoly::one();
                break;
            }
            x = y;
            let mut divisor = g.clone();
            for _ in 0..delta {
                divisor *= h.clone();
            }
            y = match r.div_int_exact(&divisor) {
                Some(q) => q,
                // Never expected (the subresultant divides exactly); falling
                // back to the primitive PRS keeps the sequence valid.
                None => r.primitive_part(),
            };
            g = x.lc();
            if delta > 0 {
                let mut gd = Integer::from(1);
                for _ in 0..delta {
                    gd *= g.clone();
                }
                let mut hd = Integer::from(1);
                for _ in 0..(delta - 1) {
                    hd *= h.clone();
                }
                h = if hd != 0 && gd.is_divisible(&hd) {
                    Integer::from(&gd / &hd)
                } else {
                    gd
                };
            }
        }
        y.primitive_part().scale_int(&d)
    }
}

/// Dense polynomial in `k` with `Z[n]` coefficients, ascending in `k`.
#[derive(Clone, Debug)]
struct BiPoly {
    c: Vec<IPoly>,
}

impl BiPoly {
    fn one() -> Self {
        BiPoly {
            c: vec![IPoly::one()],
        }
    }

    fn is_zero(&self) -> bool {
        self.c.iter().all(IPoly::is_zero)
    }

    fn degree(&self) -> i32 {
        let mut d = self.c.len() as i32 - 1;
        while d >= 0 && self.c[d as usize].is_zero() {
            d -= 1;
        }
        d
    }

    fn trim(mut self) -> Self {
        while self.c.last().map(IPoly::is_zero).unwrap_or(false) {
            self.c.pop();
        }
        self
    }

    fn lc(&self) -> IPoly {
        let d = self.degree();
        if d < 0 {
            IPoly::zero()
        } else {
            self.c[d as usize].clone()
        }
    }

    fn scale(&self, z: &IPoly) -> Self {
        if z.is_zero() {
            return BiPoly { c: vec![] };
        }
        BiPoly {
            c: self.c.iter().map(|a| a.mul(z)).collect(),
        }
        .trim()
    }

    fn sub(&self, other: &Self) -> Self {
        let n = self.c.len().max(other.c.len());
        let mut out = Vec::with_capacity(n);
        let zero = IPoly::zero();
        for i in 0..n {
            let a = self.c.get(i).unwrap_or(&zero);
            let b = other.c.get(i).unwrap_or(&zero);
            out.push(a.sub(b));
        }
        BiPoly { c: out }.trim()
    }

    /// `k^shift · self`.
    fn shift_pow(&self, shift: usize) -> Self {
        if shift == 0 || self.is_zero() {
            return self.clone();
        }
        let mut c = vec![IPoly::zero(); shift];
        c.extend(self.c.iter().cloned());
        BiPoly { c }
    }

    /// The `Z[n]`-content: gcd of the coefficients.
    fn content(&self) -> IPoly {
        let mut g = IPoly::zero();
        for a in &self.c {
            g = IPoly::gcd(&g, a);
            if g.is_one() {
                break;
            }
        }
        g
    }

    fn div_coeff_exact(&self, z: &IPoly) -> Option<Self> {
        let mut out = Vec::with_capacity(self.c.len());
        for a in &self.c {
            out.push(a.exact_div(z)?);
        }
        Some(BiPoly { c: out }.trim())
    }

    fn primitive_part(&self) -> Self {
        let cont = self.content();
        if cont.is_zero() || cont.is_one() {
            return self.clone().trim();
        }
        self.div_coeff_exact(&cont)
            .unwrap_or_else(|| self.clone().trim())
    }

    /// Exact division in `Z[n][k]`; `None` when the quotient does not live there.
    fn exact_div(&self, d: &Self) -> Option<Self> {
        if d.is_zero() {
            return None;
        }
        if self.is_zero() {
            return Some(BiPoly { c: vec![] });
        }
        let dd = d.degree();
        let da = self.degree();
        if da < dd {
            return None;
        }
        let ld = d.lc();
        let mut rem: Vec<IPoly> = self.c.clone();
        rem.resize(((da + 1) as usize).max(rem.len()), IPoly::zero());
        let mut q = vec![IPoly::zero(); (da - dd + 1) as usize];
        let mut i = da;
        while i >= dd {
            let lr = rem[i as usize].clone();
            if !lr.is_zero() {
                let t = lr.exact_div(&ld)?;
                let shift = (i - dd) as usize;
                q[shift] = t.clone();
                for (j, dc) in d.c.iter().enumerate() {
                    if dc.is_zero() {
                        continue;
                    }
                    rem[shift + j] = rem[shift + j].sub(&t.mul(dc));
                }
            }
            i -= 1;
        }
        if rem[..dd as usize].iter().any(|v| !v.is_zero()) {
            return None;
        }
        Some(BiPoly { c: q }.trim())
    }

    fn pseudo_rem(a: &Self, b: &Self) -> Self {
        let db = b.degree();
        let mut r = a.clone().trim();
        if r.is_zero() || r.degree() < db {
            return r;
        }
        let lb = b.lc();
        let mut e = (r.degree() - db + 1) as usize;
        while !r.is_zero() && r.degree() >= db {
            let shift = (r.degree() - db) as usize;
            let lr = r.lc();
            r = r.scale(&lb).sub(&b.shift_pow(shift).scale(&lr));
            e -= 1;
        }
        for _ in 0..e {
            r = r.scale(&lb);
        }
        r
    }

    /// gcd in `Z[n][k]` by Brown's subresultant PRS (content handled
    /// separately, as the algorithm requires primitive inputs).
    fn gcd(a: &Self, b: &Self) -> Self {
        let mut x = a.clone().trim();
        let mut y = b.clone().trim();
        if x.is_zero() {
            return y;
        }
        if y.is_zero() {
            return x;
        }
        if x.degree() < y.degree() {
            std::mem::swap(&mut x, &mut y);
        }
        let cx = x.content();
        let cy = y.content();
        let d = IPoly::gcd(&cx, &cy);
        x = x.div_coeff_exact(&cx).unwrap_or(x);
        y = y.div_coeff_exact(&cy).unwrap_or(y);
        if y.degree() == 0 {
            return BiPoly { c: vec![d] };
        }
        let mut g = IPoly::one();
        let mut h = IPoly::one();
        loop {
            let delta = (x.degree() - y.degree()) as usize;
            let r = BiPoly::pseudo_rem(&x, &y);
            if r.is_zero() {
                break;
            }
            if r.degree() == 0 {
                y = BiPoly::one();
                break;
            }
            x = y;
            let divisor = g.mul(&h.pow_usize(delta));
            y = match r.div_coeff_exact(&divisor) {
                Some(q) => q,
                // The subresultant divides exactly in exact arithmetic; the
                // primitive PRS is the safe fallback and stays correct.
                None => r.primitive_part(),
            };
            g = x.lc();
            if delta > 0 {
                let gd = g.pow_usize(delta);
                let hd = h.pow_usize(delta - 1);
                h = gd.exact_div(&hd).unwrap_or(gd);
            }
        }
        y.primitive_part().scale(&d)
    }
}

/// `p` rewritten as `(α / dn(n)) · B(n, k)` with `B ∈ Z[n][k]` and `α ∈ Q`.
///
/// Leaving `Q(n)` for `Z[n][k]` is what makes the gcd below cheap; the scalar
/// `α / dn` is `k`-independent, so it is a unit of `Q(n)` and cannot affect a
/// gcd or a reduced quotient — it is carried along and folded back in at the end.
struct BiForm {
    b: BiPoly,
    alpha: Rational,
    dn: RatUniPoly,
}

fn ratuni_to_ipoly(p: &RatUniPoly) -> Option<IPoly> {
    let mut c = Vec::with_capacity(p.coeffs.len());
    for q in &p.coeffs {
        if *q.clone().denom() != 1 {
            return None;
        }
        c.push(q.numer().clone());
    }
    Some(IPoly { c }.trim())
}

fn ipoly_to_ratuni(p: &IPoly) -> RatUniPoly {
    RatUniPoly {
        coeffs: p.c.iter().map(|v| Rational::from(v.clone())).collect(),
    }
    .trim()
}

fn ipoly_to_rn(p: &IPoly) -> Rn {
    RatFunc::from_poly(ipoly_to_ratuni(p))
}

/// Convert a `Q(n)[k]` polynomial into the integral-domain view.
fn to_biform(p: &PolyK) -> BiForm {
    // Common denominator in `n` across the `k`-coefficients.
    let den_of = |c: &Rn| -> RatUniPoly {
        if c.den.is_zero() {
            RatUniPoly::one()
        } else {
            c.den.clone()
        }
    };
    let mut dn = RatUniPoly::one();
    for c in &p.coeffs {
        if c.num.is_zero() {
            continue;
        }
        let den = den_of(c);
        if den.degree() <= 0 {
            continue;
        }
        let g = dn.gcd(&den);
        let (q, _) = RatUniPoly::div_rem(&den, &g);
        dn = &dn * &q;
    }
    // Numerators over that common denominator: polynomials in `n` over `Q`.
    let mut num_polys: Vec<RatUniPoly> = Vec::with_capacity(p.coeffs.len());
    for c in &p.coeffs {
        if c.num.is_zero() {
            num_polys.push(RatUniPoly::zero());
            continue;
        }
        let (mult, _) = RatUniPoly::div_rem(&dn, &den_of(c));
        num_polys.push((&c.num * &mult).trim());
    }
    // Clear the rational denominators and the integer content, so `B` has
    // integer coefficients: `Σ num_polys[i]·k^i = (cont / lcm)·B`.
    let mut den_lcm = Integer::from(1);
    for q in &num_polys {
        for c in &q.coeffs {
            den_lcm = den_lcm.lcm(&c.clone().denom().clone());
        }
    }
    let mut ints: Vec<IPoly> = Vec::with_capacity(num_polys.len());
    for q in &num_polys {
        let scaled = RatUniPoly {
            coeffs: q
                .coeffs
                .iter()
                .map(|c| c.clone() * Rational::from(den_lcm.clone()))
                .collect(),
        }
        .trim();
        // Exact by construction of `den_lcm`.
        ints.push(ratuni_to_ipoly(&scaled).unwrap_or_else(IPoly::zero));
    }
    let b = BiPoly { c: ints }.trim();
    // Integer content only — the `Z[n]` content is left in place for
    // `BiPoly::gcd`, which has to compute it anyway.
    let mut cont = Integer::from(0);
    for a in &b.c {
        cont = cont.gcd(&a.content());
        if cont == 1 {
            break;
        }
    }
    if cont == 0 {
        cont = Integer::from(1);
    }
    let b = if cont == 1 {
        b
    } else {
        let scaled: Vec<IPoly> =
            b.c.iter()
                .map(|a| a.div_int_exact(&cont).unwrap_or_else(|| a.clone()))
                .collect();
        BiPoly { c: scaled }.trim()
    };
    BiForm {
        b,
        alpha: Rational::from((cont, den_lcm)),
        dn,
    }
}

fn bipoly_to_polyk(b: &BiPoly) -> PolyK {
    PolyK {
        coeffs: b.c.iter().map(ipoly_to_rn).collect(),
    }
    .trim()
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
        let x = a.clone().trim();
        let y = b.clone().trim();
        // Cheap decisions first: they are the majority of the calls made by
        // `RatK::normalize` and by the Gosper normal form's shifted-gcd loop.
        if x.is_zero() && y.is_zero() {
            return PolyK::zero();
        }
        if y.is_zero() {
            return x.monic();
        }
        if x.is_zero() {
            return y.monic();
        }
        if x.degree() == 0 || y.degree() == 0 {
            // A non-zero constant is a unit of `Q(n)[k]`.
            return PolyK::one();
        }
        // Otherwise: subresultant PRS in `Z[n][k]` (see the module section
        // above). The `k`-independent scalars dropped by `to_biform` are units
        // of `Q(n)`, so they cannot change a monic gcd.
        let bx = to_biform(&x);
        let by = to_biform(&y);
        let g = BiPoly::gcd(&bx.b, &by.b);
        if g.degree() <= 0 {
            return PolyK::one();
        }
        bipoly_to_polyk(&g).monic()
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
        // A `k`-constant on either side is a unit of `Q(n)[k]`: there is
        // nothing to cancel, so the gcd is skipped entirely. This is the common
        // case for the small `RatK`s the search manipulates.
        if self.num.degree() > 0 && self.den.degree() > 0 {
            if let Some(reduced) = self.reduce_by_z_gcd() {
                self = reduced;
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

    /// Cancel `gcd(num, den)` without ever leaving the integral domain.
    ///
    /// The whole reduction — gcd *and* both cofactors — happens in `Z[n][k]`,
    /// so no step ever forms a quotient of rational functions in `n`. `None`
    /// means "nothing to cancel" (the gcd is a unit), which leaves the caller's
    /// representation untouched.
    fn reduce_by_z_gcd(&self) -> Option<RatK> {
        let bn = to_biform(&self.num);
        let bd = to_biform(&self.den);
        let g = BiPoly::gcd(&bn.b, &bd.b);
        if g.degree() <= 0 {
            return None;
        }
        match (bn.b.exact_div(&g), bd.b.exact_div(&g)) {
            (Some(u), Some(v)) => {
                // `num/den = (α_n/dn_n)·U / ((α_d/dn_d)·V) = s·U/V`, with the
                // `k`-independent `s = (α_n·dn_d)/(α_d·dn_n) ∈ Q(n)`.
                let s = rn_mul(
                    &rn_rat(bn.alpha / bd.alpha),
                    &RatFunc {
                        num: bd.dn,
                        den: bn.dn,
                    }
                    .normalize(),
                );
                Some(RatK {
                    num: bipoly_to_polyk(&u).scale(&s),
                    den: bipoly_to_polyk(&v),
                })
            }
            // Not expected (Gauss's lemma puts both cofactors in `Z[n][k]`);
            // dividing in `Q(n)[k]` instead is always exact here.
            _ => {
                let gk = bipoly_to_polyk(&g);
                let u = PolyK::exact_div(&self.num, &gk)?;
                let v = PolyK::exact_div(&self.den, &gk)?;
                Some(RatK { num: u, den: v })
            }
        }
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

    // -----------------------------------------------------------------
    // The `Z[n][k]` gcd against the definition it replaced
    // -----------------------------------------------------------------

    /// The Euclidean remainder sequence over the field `Q(n)` — the textbook
    /// definition of the gcd, and what [`PolyK::gcd`] used to be. Kept here as
    /// the oracle the fast subresultant version is checked against: the two
    /// must agree on the nose, since a monic gcd is unique.
    fn gcd_by_field_euclid(a: &PolyK, b: &PolyK) -> PolyK {
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

    /// A cheap deterministic stream, so the cases below are reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 33
        }

        /// A small integer in `[-3, 3]`.
        fn small(&mut self) -> i64 {
            (self.next() % 7) as i64 - 3
        }

        /// A `Q(n)`-coefficient polynomial in `k` of degree `< deg`, with
        /// coefficients that are genuine rational functions of `n`.
        fn poly(&mut self, deg: usize) -> PolyK {
            let mut coeffs = Vec::with_capacity(deg);
            for _ in 0..deg {
                let num = rn_of(&[self.small(), self.small(), self.small()]);
                let den = rn_of(&[self.small(), self.small()]);
                coeffs.push(if rn_is_zero(&den) {
                    num
                } else {
                    rn_div(&num, &den).unwrap()
                });
            }
            PolyK::from_coeffs(coeffs)
        }
    }

    #[test]
    fn gcd_agrees_with_the_field_euclidean_sequence() {
        let mut rng = Lcg(0x5eed_1234);
        for _ in 0..40 {
            let a = rng.poly(3);
            let b = rng.poly(3);
            // Random pairs (usually coprime) *and* pairs with a planted factor,
            // so both the "nothing cancels" and the "something cancels" branch
            // are exercised.
            let g = rng.poly(2);
            for (x, y) in [(a.clone(), b.clone()), (a.mul(&g), b.mul(&g))] {
                let fast = PolyK::gcd(&x, &y);
                let slow = gcd_by_field_euclid(&x, &y);
                assert!(
                    fast.eq_poly(&slow),
                    "gcd mismatch:\n  x = {x:?}\n  y = {y:?}\n  fast = {fast:?}\n  slow = {slow:?}"
                );
            }
        }
    }

    #[test]
    fn gcd_edge_cases_match_the_old_contract() {
        let z = PolyK::zero();
        let one = PolyK::one();
        let p = PolyK::from_coeffs(vec![rn_var(), rn_one()]); // k + n
        assert!(PolyK::gcd(&z, &z).is_zero());
        assert!(PolyK::gcd(&p, &z).eq_poly(&p.monic()));
        assert!(PolyK::gcd(&z, &p).eq_poly(&p.monic()));
        assert!(PolyK::gcd(&p, &one).eq_poly(&one));
        assert!(PolyK::gcd(&p, &p).eq_poly(&p.monic()));
    }

    #[test]
    fn normalize_cancels_a_planted_common_factor_exactly() {
        let mut rng = Lcg(0xfeed_beef);
        for _ in 0..20 {
            let u = rng.poly(3);
            let v = rng.poly(3);
            let g = rng.poly(3);
            if u.is_zero() || v.is_zero() || g.is_zero() {
                continue;
            }
            let r = RatK {
                num: u.mul(&g),
                den: v.mul(&g),
            }
            .normalize();
            // Reduced: nothing left to cancel …
            assert_eq!(
                PolyK::gcd(&r.num, &r.den).degree().max(0),
                0,
                "normalize must leave a reduced fraction"
            );
            // … denominator monic …
            assert!(rn_eq(&r.den.leading_coeff(), &rn_one()));
            // … and still the same element of `Q(n)(k)`.
            let want = RatK {
                num: u.clone(),
                den: v.clone(),
            }
            .normalize();
            assert!(r.eq_ratk(&want));
            assert!(r.num.eq_poly(&want.num) && r.den.eq_poly(&want.den));
        }
    }

    #[test]
    fn rn_arithmetic_matches_the_generic_ratfunc_operators() {
        // `rn_add`/`rn_mul`/`rn_inv` cancel crosswise instead of reducing the
        // full product; the representative they produce must be the same one
        // `RatFunc`'s own (fully normalizing) operators produce.
        let mut rng = Lcg(0x1234_5678);
        for _ in 0..200 {
            let mk = |g: &mut Lcg| {
                let num = rn_of(&[g.small(), g.small(), g.small()]);
                let den = rn_of(&[g.small(), g.small()]);
                if rn_is_zero(&den) {
                    num
                } else {
                    rn_div(&num, &den).unwrap()
                }
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            assert_eq!(rn_add(&a, &b), a.clone() + b.clone());
            assert_eq!(rn_mul(&a, &b), a.mul_ratfunc(&b));
            assert_eq!(rn_inv(&a), a.inv());
        }
    }

    #[test]
    fn q_gcd_matches_the_naive_euclidean_gcd_over_q() {
        let mut rng = Lcg(0x0bad_c0de);
        for _ in 0..100 {
            let mk = |g: &mut Lcg| {
                RatUniPoly {
                    coeffs: (0..4)
                        .map(|_| Rational::from((g.small(), 2)))
                        .collect::<Vec<_>>(),
                }
                .trim()
            };
            let a = mk(&mut rng);
            let b = mk(&mut rng);
            if a.is_zero() || b.is_zero() {
                continue;
            }
            assert_eq!(q_gcd(&a, &b), a.gcd(&b), "a = {a:?}, b = {b:?}");
            let c = &a * &b;
            assert_eq!(q_gcd(&c, &a), c.gcd(&a));
        }
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
