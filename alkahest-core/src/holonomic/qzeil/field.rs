//! The coefficient tower `q`-Zeilberger runs in: `Q(q) ⊂ Q(q)(x) ⊂ Q(q)(x)(y)`.
//!
//! With `x = qⁿ` and `y = q^k`, every shift quotient of a `q`-hypergeometric
//! term is a *rational function* of `x` and `y` over `Q(q)`, and the two shift
//! operators act on that field as the `Q(q)`-algebra maps
//!
//! ```text
//! n ↦ n+1   is   x ↦ q·x        k ↦ k+1   is   y ↦ q·y
//! ```
//!
//! which is what makes the classical Gosper/Zeilberger derivation carry over
//! verbatim with `k+1` read as `q·y` — see [`super::search`].
//!
//! # Reuse, deliberately
//!
//! The bottom two levels are **not** new code. [`super::super::qfield`]'s
//! [`Rn`] is `Q(v)` for one variable `v` and [`PolyK`]/[`RatK`] are `Rn[v₂]`
//! and `Rn(v₂)`; naming the outer variable `q` instead of `n` and the inner one
//! `x` instead of `k` gives exactly `Q(q)`, `Q(q)[x]` and `Q(q)(x)`, with the
//! same tested arithmetic (including the `Z[·][·]` gcd that keeps `normalize`
//! from blowing up). The aliases [`Qq`], [`PolyX`], [`RatX`] record that
//! reinterpretation.
//!
//! What is *not* reused is the shift: `PolyK::shift_k` / `shift_n` are additive
//! (`v ↦ v + j`), and the `q`-world needs the multiplicative `x ↦ q^i·x`, which
//! is [`polyx_qshift`] / [`ratx_qshift`] here. Only the third level, `y`, is
//! new: [`PolyY`] and [`RatY`] below.

use super::super::qfield::{
    rn_add, rn_inv, rn_is_zero, rn_mul, rn_one, rn_poly, rn_zero, PolyK, RatK, Rn,
};
use crate::matrix::normal_form::RatUniPoly;
use rug::Rational;

/// `Q(q)` — rational functions in the `q` of a `q`-analogue.
pub type Qq = Rn;
/// `Q(q)[x]` with `x = qⁿ`.
pub type PolyX = PolyK;
/// `Q(q)(x)` — where the recurrence coefficients `a_i` live.
pub type RatX = RatK;

/// `q^i ∈ Q(q)`, for any sign of `i`.
pub fn qq_pow(i: i64) -> Qq {
    if i == 0 {
        return rn_one();
    }
    let mut coeffs = vec![Rational::from(0); i.unsigned_abs() as usize + 1];
    coeffs[i.unsigned_abs() as usize] = Rational::from(1);
    let mono = rn_poly(RatUniPoly { coeffs }.trim());
    if i > 0 {
        mono
    } else {
        // `mono` is the monomial `q^{|i|} ≠ 0`, so the inverse exists.
        rn_inv(&mono).unwrap_or_else(rn_one)
    }
}

/// `x^a ∈ Q(q)(x)`, for any sign of `a`.
pub fn ratx_x_pow(a: i64) -> RatX {
    let mut coeffs = vec![rn_zero(); a.unsigned_abs() as usize + 1];
    coeffs[a.unsigned_abs() as usize] = rn_one();
    let mono = RatX::from_poly(PolyX::from_coeffs(coeffs));
    if a >= 0 {
        mono
    } else {
        mono.inv().unwrap_or_else(RatX::one)
    }
}

/// `p(x)` with `x ↦ q^i·x` — the action of `n ↦ n+i` on `Q(q)[x]`.
pub fn polyx_qshift(p: &PolyX, i: i64) -> PolyX {
    if i == 0 {
        return p.clone();
    }
    PolyX::from_coeffs(
        p.coeffs
            .iter()
            .enumerate()
            .map(|(d, c)| rn_mul(c, &qq_pow(i * d as i64)))
            .collect(),
    )
}

/// `r(x)` with `x ↦ q^i·x` — the action of `n ↦ n+i` on `Q(q)(x)`.
pub fn ratx_qshift(r: &RatX, i: i64) -> RatX {
    if i == 0 {
        return r.clone();
    }
    RatX {
        num: polyx_qshift(&r.num, i),
        den: polyx_qshift(&r.den, i),
    }
    .normalize()
}

// ---------------------------------------------------------------------------
// Q(q)(x)[y]
// ---------------------------------------------------------------------------

/// A polynomial in `y = q^k` with coefficients in `Q(q)(x)` (ascending order).
///
/// The same dense representation as [`PolyK`], one level up the tower; the
/// operations are the field-generic ones, so the coefficient arithmetic is
/// `RatX`'s and every reduction below is exact.
#[derive(Clone, Debug)]
pub struct PolyY {
    pub coeffs: Vec<RatX>,
}

impl PolyY {
    pub fn zero() -> Self {
        PolyY { coeffs: vec![] }
    }

    pub fn one() -> Self {
        PolyY {
            coeffs: vec![RatX::one()],
        }
    }

    pub fn constant(c: RatX) -> Self {
        PolyY { coeffs: vec![c] }.trim()
    }

    /// The polynomial `y`.
    pub fn y() -> Self {
        PolyY {
            coeffs: vec![RatX::zero(), RatX::one()],
        }
    }

    pub fn from_coeffs(coeffs: Vec<RatX>) -> Self {
        PolyY { coeffs }.trim()
    }

    pub fn trim(mut self) -> Self {
        while self.coeffs.last().map(RatX::is_zero).unwrap_or(false) {
            self.coeffs.pop();
        }
        self
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(RatX::is_zero)
    }

    /// Degree, or `-1` for the zero polynomial.
    pub fn degree(&self) -> i32 {
        let mut d = self.coeffs.len() as i32 - 1;
        while d >= 0 && self.coeffs[d as usize].is_zero() {
            d -= 1;
        }
        d
    }

    pub fn coeff(&self, i: usize) -> RatX {
        self.coeffs.get(i).cloned().unwrap_or_else(RatX::zero)
    }

    pub fn leading_coeff(&self) -> RatX {
        let d = self.degree();
        if d < 0 {
            RatX::zero()
        } else {
            self.coeff(d as usize)
        }
    }

    pub fn add(&self, other: &PolyY) -> PolyY {
        let n = self.coeffs.len().max(other.coeffs.len());
        PolyY {
            coeffs: (0..n).map(|i| self.coeff(i).add(&other.coeff(i))).collect(),
        }
        .trim()
    }

    pub fn neg(&self) -> PolyY {
        PolyY {
            coeffs: self.coeffs.iter().map(RatX::neg).collect(),
        }
    }

    pub fn sub(&self, other: &PolyY) -> PolyY {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &PolyY) -> PolyY {
        if self.is_zero() || other.is_zero() {
            return PolyY::zero();
        }
        let mut out = vec![RatX::zero(); self.coeffs.len() + other.coeffs.len() - 1];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                if b.is_zero() {
                    continue;
                }
                out[i + j] = out[i + j].add(&a.mul(b));
            }
        }
        PolyY { coeffs: out }.trim()
    }

    pub fn scale(&self, c: &RatX) -> PolyY {
        if c.is_zero() {
            return PolyY::zero();
        }
        PolyY {
            coeffs: self.coeffs.iter().map(|a| a.mul(c)).collect(),
        }
        .trim()
    }

    /// Euclidean division over the field `Q(q)(x)`.
    pub fn div_rem(a: &PolyY, b: &PolyY) -> Option<(PolyY, PolyY)> {
        if b.is_zero() {
            return None;
        }
        let db = b.degree();
        let lb_inv = b.leading_coeff().inv()?;
        let mut rem = a.clone().trim();
        let mut quot: Vec<RatX> = Vec::new();
        while !rem.is_zero() && rem.degree() >= db {
            let shift = (rem.degree() - db) as usize;
            let t = rem.leading_coeff().mul(&lb_inv);
            if shift >= quot.len() {
                quot.resize(shift + 1, RatX::zero());
            }
            quot[shift] = quot[shift].add(&t);
            let mut sub_coeffs = vec![RatX::zero(); shift];
            sub_coeffs.extend(b.coeffs.iter().map(|c| c.mul(&t)));
            rem = rem.sub(&PolyY { coeffs: sub_coeffs });
        }
        Some((PolyY { coeffs: quot }.trim(), rem.trim()))
    }

    pub fn exact_div(a: &PolyY, b: &PolyY) -> Option<PolyY> {
        let (q, r) = PolyY::div_rem(a, b)?;
        r.is_zero().then_some(q)
    }

    /// Monic gcd over `Q(q)(x)`, by the Euclidean algorithm.
    pub fn gcd(a: &PolyY, b: &PolyY) -> PolyY {
        let mut x = a.clone().trim();
        let mut y = b.clone().trim();
        if x.is_zero() && y.is_zero() {
            return PolyY::zero();
        }
        while !y.is_zero() {
            let Some((_, r)) = PolyY::div_rem(&x, &y) else {
                return PolyY::one();
            };
            x = y;
            y = r;
        }
        x.monic()
    }

    pub fn monic(&self) -> PolyY {
        match self.leading_coeff().inv() {
            Some(inv) => self.scale(&inv),
            None => self.clone(),
        }
    }

    /// `lcm` via `a·b/gcd`.
    pub fn lcm(a: &PolyY, b: &PolyY) -> PolyY {
        if a.is_zero() || b.is_zero() {
            return PolyY::zero();
        }
        let g = PolyY::gcd(a, b);
        let prod = a.mul(b);
        PolyY::exact_div(&prod, &g).unwrap_or(prod)
    }

    /// `p` with `y ↦ q^j·y` — the action of `k ↦ k+j`.
    pub fn qshift_y(&self, j: i64) -> PolyY {
        if j == 0 {
            return self.clone();
        }
        PolyY::from_coeffs(
            self.coeffs
                .iter()
                .enumerate()
                .map(|(d, c)| c.mul(&RatX::from_rn(qq_pow(j * d as i64))))
                .collect(),
        )
    }

    /// `p` with `x ↦ q^i·x` — the action of `n ↦ n+i`.
    pub fn qshift_x(&self, i: i64) -> PolyY {
        if i == 0 {
            return self.clone();
        }
        PolyY::from_coeffs(self.coeffs.iter().map(|c| ratx_qshift(c, i)).collect())
    }

    pub fn eq_poly(&self, other: &PolyY) -> bool {
        self.sub(other).is_zero()
    }
}

// ---------------------------------------------------------------------------
// Q(q)(x)(y)
// ---------------------------------------------------------------------------

/// A rational function in `y` over `Q(q)(x)` — where shift quotients and the
/// certificate live, and where the final identity is checked.
#[derive(Clone, Debug)]
pub struct RatY {
    pub num: PolyY,
    pub den: PolyY,
}

impl RatY {
    pub fn zero() -> Self {
        RatY {
            num: PolyY::zero(),
            den: PolyY::one(),
        }
    }

    pub fn one() -> Self {
        RatY {
            num: PolyY::one(),
            den: PolyY::one(),
        }
    }

    pub fn from_poly(p: PolyY) -> Self {
        RatY {
            num: p,
            den: PolyY::one(),
        }
        .normalize()
    }

    pub fn from_ratx(c: RatX) -> Self {
        RatY::from_poly(PolyY::constant(c))
    }

    pub fn y() -> Self {
        RatY::from_poly(PolyY::y())
    }

    pub fn is_zero(&self) -> bool {
        self.num.is_zero()
    }

    pub fn normalize(mut self) -> Self {
        if self.num.is_zero() {
            return RatY::zero();
        }
        if self.den.is_zero() {
            return self;
        }
        if self.num.degree() > 0 && self.den.degree() > 0 {
            let g = PolyY::gcd(&self.num, &self.den);
            if g.degree() > 0 {
                if let (Some(u), Some(v)) = (
                    PolyY::exact_div(&self.num, &g),
                    PolyY::exact_div(&self.den, &g),
                ) {
                    self.num = u;
                    self.den = v;
                }
            }
        }
        if let Some(inv) = self.den.leading_coeff().inv() {
            self.num = self.num.scale(&inv);
            self.den = self.den.scale(&inv);
        }
        self
    }

    pub fn add(&self, other: &RatY) -> RatY {
        RatY {
            num: self.num.mul(&other.den).add(&other.num.mul(&self.den)),
            den: self.den.mul(&other.den),
        }
        .normalize()
    }

    pub fn neg(&self) -> RatY {
        RatY {
            num: self.num.neg(),
            den: self.den.clone(),
        }
    }

    pub fn sub(&self, other: &RatY) -> RatY {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &RatY) -> RatY {
        RatY {
            num: self.num.mul(&other.num),
            den: self.den.mul(&other.den),
        }
        .normalize()
    }

    pub fn inv(&self) -> Option<RatY> {
        if self.num.is_zero() {
            return None;
        }
        Some(
            RatY {
                num: self.den.clone(),
                den: self.num.clone(),
            }
            .normalize(),
        )
    }

    pub fn div(&self, other: &RatY) -> Option<RatY> {
        Some(self.mul(&other.inv()?))
    }

    pub fn pow_i32(&self, e: i32) -> Option<RatY> {
        if e == 0 {
            return Some(RatY::one());
        }
        let base = if e < 0 { self.inv()? } else { self.clone() };
        let mut acc = RatY::one();
        for _ in 0..e.unsigned_abs() {
            acc = acc.mul(&base);
        }
        Some(acc)
    }

    /// `r` with `y ↦ q^j·y` — the action of `k ↦ k+j`.
    pub fn qshift_y(&self, j: i64) -> RatY {
        RatY {
            num: self.num.qshift_y(j),
            den: self.den.qshift_y(j),
        }
        .normalize()
    }

    /// `r` with `x ↦ q^i·x` — the action of `n ↦ n+i`.
    pub fn qshift_x(&self, i: i64) -> RatY {
        RatY {
            num: self.num.qshift_x(i),
            den: self.den.qshift_x(i),
        }
        .normalize()
    }

    pub fn eq_raty(&self, other: &RatY) -> bool {
        self.sub(other).is_zero()
    }
}

/// `x^a·y^b·q^c` as an element of `Q(q)(x)(y)` — the image of `q^{a·n + b·k + c}`.
pub fn q_monomial(a: i64, b: i64, c: i64) -> RatY {
    let scalar = RatX::from_rn(qq_pow(c)).mul(&ratx_x_pow(a));
    let mut out = RatY::from_ratx(scalar);
    if b != 0 {
        let mut coeffs = vec![RatX::zero(); b.unsigned_abs() as usize + 1];
        coeffs[b.unsigned_abs() as usize] = RatX::one();
        let mono = RatY::from_poly(PolyY::from_coeffs(coeffs));
        let mono = if b > 0 {
            mono
        } else {
            // A monomial is never zero, so the inverse exists.
            mono.inv().unwrap_or_else(RatY::one)
        };
        out = out.mul(&mono);
    }
    out
}

// ---------------------------------------------------------------------------
// Evaluation at integer (n, k)
// ---------------------------------------------------------------------------

/// `p` at `x = q^{n₀}`.
pub fn polyx_at_qn(p: &PolyX, n0: i64) -> Qq {
    let mut acc = rn_zero();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if rn_is_zero(c) {
            continue;
        }
        acc = rn_add(&acc, &rn_mul(c, &qq_pow(n0 * deg as i64)));
    }
    acc
}

/// `r` at `x = q^{n₀}`, or `None` at a pole.
pub fn ratx_at_qn(r: &RatX, n0: i64) -> Option<Qq> {
    let den = polyx_at_qn(&r.den, n0);
    if rn_is_zero(&den) {
        return None;
    }
    Some(rn_mul(&polyx_at_qn(&r.num, n0), &rn_inv(&den)?))
}

/// `p` at `x = q^{n₀}`, `y = q^{k₀}`, or `None` at a coefficient pole.
pub fn polyy_at(p: &PolyY, n0: i64, k0: i64) -> Option<Qq> {
    let mut acc = rn_zero();
    for (deg, c) in p.coeffs.iter().enumerate() {
        if c.is_zero() {
            continue;
        }
        let cv = ratx_at_qn(c, n0)?;
        acc = rn_add(&acc, &rn_mul(&cv, &qq_pow(k0 * deg as i64)));
    }
    Some(acc)
}

/// `r` at `x = q^{n₀}`, `y = q^{k₀}`, or `None` at a pole.
pub fn raty_at(r: &RatY, n0: i64, k0: i64) -> Option<Qq> {
    let den = polyy_at(&r.den, n0, k0)?;
    if rn_is_zero(&den) {
        return None;
    }
    Some(rn_mul(&polyy_at(&r.num, n0, k0)?, &rn_inv(&den)?))
}

/// Clear the denominators of a `Q(q)(x)` family, returning the numerators over
/// one common `Q(q)[x]` scale.
///
/// The scale is `k`-free, so multiplying the certificate by the same factor
/// preserves the telescoping identity exactly — which is why the search may do
/// it *before* the verification step rather than after.
pub fn clear_denominators_x(items: &[RatX]) -> (Vec<PolyX>, RatX) {
    let mut den = PolyX::one();
    for it in items {
        if it.is_zero() {
            continue;
        }
        den = PolyX::lcm(&den, &it.den);
    }
    let scale = RatX::from_poly(den);
    let out = items
        .iter()
        .map(|it| {
            let prod = it.mul(&scale);
            // Exact by construction of the lcm; fall back to the numerator
            // rather than panicking if a degenerate denominator slipped in.
            if prod.den.degree() == 0 {
                let c = prod.den.coeff(0);
                match rn_inv(&c) {
                    Some(inv) => prod.num.scale(&inv),
                    None => prod.num.clone(),
                }
            } else {
                prod.num.clone()
            }
        })
        .collect();
    (out, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qq_pow_is_a_group_homomorphism() {
        for i in -3_i64..4 {
            for j in -3_i64..4 {
                let lhs = rn_mul(&qq_pow(i), &qq_pow(j));
                let rhs = qq_pow(i + j);
                assert!(
                    super::super::super::qfield::rn_eq(&lhs, &rhs),
                    "q^{i}·q^{j} must be q^{}",
                    i + j
                );
            }
        }
    }

    #[test]
    fn y_shift_is_multiplicative_on_monomials() {
        // (y²)|_{y ↦ q y} = q²·y².
        let y2 = PolyY::y().mul(&PolyY::y());
        let shifted = y2.qshift_y(1);
        let expect = y2.scale(&RatX::from_rn(qq_pow(2)));
        assert!(shifted.eq_poly(&expect));
    }

    #[test]
    fn x_shift_acts_on_x_powers() {
        // x|_{x ↦ q^3 x} = q³·x.
        let x = ratx_x_pow(1);
        let shifted = ratx_qshift(&x, 3);
        let expect = x.mul(&RatX::from_rn(qq_pow(3)));
        assert!(shifted.eq_ratk(&expect));
    }

    #[test]
    fn raty_is_a_field_on_a_small_sample() {
        let a = RatY::from_poly(PolyY::from_coeffs(vec![
            RatX::one(),
            ratx_x_pow(1),
            RatX::from_rn(qq_pow(2)),
        ]));
        let b = RatY::from_poly(PolyY::from_coeffs(vec![ratx_x_pow(-1), RatX::one()]));
        let q = a.div(&b).expect("b != 0");
        assert!(q.mul(&b).eq_raty(&a), "division must invert multiplication");
        assert!(a.sub(&a).is_zero());
    }

    #[test]
    fn q_monomial_matches_repeated_multiplication() {
        let m = q_monomial(2, -1, 3);
        let expect = RatY::from_ratx(ratx_x_pow(2).mul(&RatX::from_rn(qq_pow(3))))
            .div(&RatY::y())
            .expect("y != 0");
        assert!(m.eq_raty(&expect));
    }

    #[test]
    fn gcd_and_exact_div_agree() {
        let f = PolyY::from_coeffs(vec![RatX::one(), RatX::one()]); // y + 1
        let g = PolyY::from_coeffs(vec![ratx_x_pow(1), RatX::one()]); // y + x
        let prod = f.mul(&g);
        let d = PolyY::gcd(&prod, &f);
        assert_eq!(d.degree(), 1);
        assert!(PolyY::exact_div(&prod, &f).expect("divides").eq_poly(&g));
    }
}
