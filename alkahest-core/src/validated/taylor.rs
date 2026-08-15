//! Taylor model arithmetic with rigorous remainder bounds.
//!
//! A [`TaylorModel`] over an `n`-dimensional box represents a function `f` as
//! a truncated polynomial `P` in **normalised** coordinates `u ∈ [-1,1]ⁿ`
//! plus an interval remainder `I`:
//!
//! ```text
//! x = c + r ⊙ u,      ∀ u ∈ [-1,1]ⁿ :  f(x) ∈ P(u) + I
//! ```
//!
//! Normalised coordinates make range bounding trivial and numerically stable:
//! every monomial `u^α` lies in `[-1,1]`, and in `[0,1]` when every exponent
//! in `α` is even.
//!
//! Every operation preserves the enclosure property.  Terms above the
//! truncation order are bounded over `[-1,1]ⁿ` and folded into `I`; they are
//! never discarded.  Elementary functions use a Lagrange remainder whose
//! derivative bound is evaluated over an enclosure of the entire argument
//! range.

use super::{
    contains_zero, from_bounds, from_float, hull, is_finite, lb, mag, mig, pi_ball, symmetric, ub,
    ValidatedError,
};
use crate::ball::ArbBall;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{Complete, Float, Integer};
use std::collections::{BTreeMap, HashMap};

type Result<T> = std::result::Result<T, ValidatedError>;

/// Exponent vector of a monomial, one entry per box variable.
pub type MultiIndex = Vec<u32>;

/// Highest Taylor order accepted.  Beyond this the coefficient count and the
/// `(p+1)!` remainder scaling stop buying accuracy.
pub const MAX_ORDER: usize = 24;

/// `v > 0`, with NaN answering **false**.
///
/// The guards below are all of the form "refuse unless strictly positive", and
/// that has to keep refusing on a NaN endpoint. Writing them as `v <= 0` would
/// invert exactly that case and let a NaN through into a certificate, so the
/// comparison is spelled out through `partial_cmp`.
fn strictly_positive(v: &Float) -> bool {
    matches!(v.partial_cmp(&0), Some(std::cmp::Ordering::Greater))
}

/// A Taylor model: polynomial part with ball coefficients plus a rigorously
/// enclosing remainder interval.
#[derive(Clone, Debug)]
pub struct TaylorModel {
    nvars: usize,
    order: usize,
    prec: u32,
    /// Coefficients of `P` keyed by exponent vector in normalised coordinates.
    coeffs: BTreeMap<MultiIndex, ArbBall>,
    /// Interval remainder — an enclosure of `f - P` over the whole box.
    remainder: ArbBall,
}

impl TaylorModel {
    // ── constructors ─────────────────────────────────────────────────────

    /// The zero model.
    pub fn zero(nvars: usize, order: usize, prec: u32) -> Self {
        TaylorModel {
            nvars,
            order,
            prec,
            coeffs: BTreeMap::new(),
            remainder: ArbBall::from_f64(0.0, prec),
        }
    }

    /// A constant model enclosing the ball `c`.
    pub fn constant(c: ArbBall, nvars: usize, order: usize, prec: u32) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(vec![0u32; nvars], c);
        TaylorModel {
            nvars,
            order,
            prec,
            coeffs,
            remainder: ArbBall::from_f64(0.0, prec),
        }
    }

    /// The model of the `i`-th coordinate over a box whose `i`-th interval has
    /// midpoint `center` and radius `radius`: `xᵢ = center + radius · uᵢ`.
    pub fn variable(
        i: usize,
        center: &Float,
        radius: &Float,
        nvars: usize,
        order: usize,
        prec: u32,
    ) -> Self {
        let mut coeffs = BTreeMap::new();
        coeffs.insert(vec![0u32; nvars], from_float(center, prec));
        if order >= 1 {
            let mut e = vec![0u32; nvars];
            e[i] = 1;
            coeffs.insert(e, from_float(radius, prec));
        }
        TaylorModel {
            nvars,
            order,
            prec,
            coeffs,
            // With order >= 1 the linear term is exact, so no remainder is
            // needed.  With order 0 the linear part must be absorbed.
            remainder: if order >= 1 {
                ArbBall::from_f64(0.0, prec)
            } else {
                symmetric(radius, prec)
            },
        }
    }

    // ── accessors ────────────────────────────────────────────────────────

    /// Number of box variables.
    pub fn nvars(&self) -> usize {
        self.nvars
    }

    /// Truncation order (maximum total degree kept in `P`).
    pub fn order(&self) -> usize {
        self.order
    }

    /// Working precision in bits.
    pub fn prec(&self) -> u32 {
        self.prec
    }

    /// The remainder interval `I`.
    pub fn remainder(&self) -> &ArbBall {
        &self.remainder
    }

    /// Number of retained polynomial terms.
    pub fn term_count(&self) -> usize {
        self.coeffs.len()
    }

    fn zero_index(&self) -> MultiIndex {
        vec![0u32; self.nvars]
    }

    fn coeff(&self, idx: &MultiIndex) -> ArbBall {
        self.coeffs
            .get(idx)
            .cloned()
            .unwrap_or_else(|| ArbBall::from_f64(0.0, self.prec))
    }

    fn insert(&mut self, idx: MultiIndex, c: ArbBall) {
        match self.coeffs.get_mut(&idx) {
            Some(existing) => *existing = existing.clone() + c,
            None => {
                self.coeffs.insert(idx, c);
            }
        }
    }

    // ── range bounding ───────────────────────────────────────────────────

    /// Rigorous enclosure of the polynomial part over `[-1,1]ⁿ`.
    ///
    /// The constant term is kept centred; a monomial with at least one odd
    /// exponent contributes `[-|c|, |c|]`; a monomial whose exponents are all
    /// even contributes `hull(0, c)` because `u^α ∈ [0,1]` there.
    pub fn poly_bound(&self) -> ArbBall {
        let zero = ArbBall::from_f64(0.0, self.prec);
        let mut acc = zero.clone();
        for (idx, c) in &self.coeffs {
            if idx.iter().all(|&e| e == 0) {
                acc = acc + c.clone();
            } else if idx.iter().all(|&e| e % 2 == 0) {
                acc = acc + hull(&zero, c);
            } else {
                acc = acc + symmetric(&mag(c), self.prec);
            }
        }
        acc
    }

    /// Rigorous enclosure of `f` over the whole box: `P([-1,1]ⁿ) + I`.
    pub fn range(&self) -> ArbBall {
        self.poly_bound() + self.remainder.clone()
    }

    fn check_finite(&self, what: &str) -> Result<()> {
        if !is_finite(&self.remainder) || self.coeffs.values().any(|c| !is_finite(c)) {
            return Err(ValidatedError::NotFinite {
                what: what.to_string(),
            });
        }
        Ok(())
    }

    // ── arithmetic ───────────────────────────────────────────────────────

    /// Negation.
    pub fn neg(&self) -> Self {
        let mut out = self.clone();
        out.coeffs = self
            .coeffs
            .iter()
            .map(|(k, v)| (k.clone(), -v.clone()))
            .collect();
        out.remainder = -self.remainder.clone();
        out
    }

    /// Addition.
    pub fn add(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for (idx, c) in &other.coeffs {
            out.insert(idx.clone(), c.clone());
        }
        out.remainder = out.remainder + other.remainder.clone();
        out
    }

    /// Subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Multiply by a ball constant.
    pub fn scale(&self, k: &ArbBall) -> Self {
        let mut out = self.clone();
        out.coeffs = self
            .coeffs
            .iter()
            .map(|(i, c)| (i.clone(), c.clone() * k.clone()))
            .collect();
        out.remainder = self.remainder.clone() * k.clone();
        out
    }

    /// Add a ball constant.
    pub fn shift(&self, k: &ArbBall) -> Self {
        let mut out = self.clone();
        let z = out.zero_index();
        out.insert(z, k.clone());
        out
    }

    /// Multiplication.  Products of total degree above the truncation order
    /// are bounded over `[-1,1]ⁿ` and folded into the remainder.
    pub fn mul(&self, other: &Self) -> Self {
        let prec = self.prec.max(other.prec);
        let order = self.order.min(other.order);
        let zero = ArbBall::from_f64(0.0, prec);
        let mut out = TaylorModel::zero(self.nvars, order, prec);
        let mut truncated = zero.clone();

        for (a_idx, a) in &self.coeffs {
            for (b_idx, b) in &other.coeffs {
                let deg: u32 = a_idx.iter().zip(b_idx).map(|(x, y)| x + y).sum();
                let prod = a.clone() * b.clone();
                if deg as usize <= order {
                    let idx: MultiIndex = a_idx.iter().zip(b_idx).map(|(x, y)| x + y).collect();
                    out.insert(idx, prod);
                } else {
                    // The dropped monomial lies in [-1,1] (or [0,1] when all
                    // exponents are even); bound it and keep it.
                    let all_even = a_idx
                        .iter()
                        .zip(b_idx)
                        .all(|(x, y)| (x + y) % 2 == 0 && (x + y) > 0);
                    if all_even {
                        truncated = truncated + hull(&zero, &prod);
                    } else {
                        truncated = truncated + symmetric(&mag(&prod), prec);
                    }
                }
            }
        }

        // f ∈ P₁ + I₁, g ∈ P₂ + I₂  ⟹  fg ∈ P₁P₂ + P₁I₂ + P₂I₁ + I₁I₂.
        let pb_a = self.poly_bound();
        let pb_b = other.poly_bound();
        out.remainder = truncated
            + pb_a * other.remainder.clone()
            + pb_b * self.remainder.clone()
            + self.remainder.clone() * other.remainder.clone();
        out
    }

    /// Non-negative integer power by binary exponentiation.
    pub fn powi_nonneg(&self, n: u32) -> Self {
        if n == 0 {
            return TaylorModel::constant(
                ArbBall::from_f64(1.0, self.prec),
                self.nvars,
                self.order,
                self.prec,
            );
        }
        let mut result: Option<Self> = None;
        let mut base = self.clone();
        let mut e = n;
        while e > 0 {
            if e & 1 == 1 {
                result = Some(match result {
                    Some(r) => r.mul(&base),
                    None => base.clone(),
                });
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base);
            }
        }
        result.unwrap()
    }

    /// Integer power, including negative exponents (via [`Self::recip`]).
    pub fn powi(&self, n: i64) -> Result<Self> {
        if n >= 0 {
            let n: u32 = u32::try_from(n).map_err(|_| ValidatedError::Unsupported {
                what: format!("integer exponent {n} is too large for a Taylor model"),
            })?;
            Ok(self.powi_nonneg(n))
        } else {
            let p =
                self.powi_nonneg(u32::try_from(-n).map_err(|_| ValidatedError::Unsupported {
                    what: format!("integer exponent {n} is too large for a Taylor model"),
                })?);
            p.recip()
        }
    }

    /// Division.  Refuses when the divisor's range contains zero.
    pub fn div(&self, other: &Self) -> Result<Self> {
        Ok(self.mul(&other.recip()?))
    }

    // ── composition machinery ────────────────────────────────────────────

    /// Split off the (exact) midpoint of the constant term.
    ///
    /// Returns `(m₀, Δ)` with `Δ = self - m₀`, so `self = m₀ + Δ` and the
    /// expansion point `m₀` is a genuine point rather than a ball.
    fn center_split(&self) -> (Float, Self) {
        let m0 = self.coeff(&self.zero_index()).mid.clone();
        let shifted = self.shift(&(-from_float(&m0, self.prec)));
        (m0, shifted)
    }

    /// Compose with a univariate function given its Taylor coefficients
    /// `a₀..a_p` at the expansion point and a bound on the degree-`(p+1)`
    /// Lagrange remainder.
    ///
    /// `self` must be the *centred* part `Δ` and `rem_radius` must bound
    /// `|f^{(p+1)}(ξ)| / (p+1)! · |Δ|^{p+1}` uniformly over the box.
    fn compose(&self, a: &[ArbBall], rem_radius: &Float) -> Self {
        debug_assert_eq!(a.len(), self.order + 1);
        let mut acc =
            TaylorModel::constant(a[self.order].clone(), self.nvars, self.order, self.prec);
        for k in (0..self.order).rev() {
            acc = acc.mul(self).shift(&a[k]);
        }
        acc.remainder = acc.remainder + symmetric(rem_radius, self.prec);
        acc
    }

    /// `|Δ|^{p+1}` as a rounded-up `Float`, where `Δ` is `self`.
    fn delta_pow(&self, d_range: &ArbBall) -> Float {
        let prec = self.prec;
        let m = mag(d_range);
        let b = ArbBall {
            mid: m,
            rad: Float::new(prec),
            prec,
        };
        ub(&b.powi((self.order + 1) as i64))
    }

    fn factorial(k: usize, prec: u32) -> ArbBall {
        let f = Integer::factorial(k as u32).complete();
        ArbBall::from_integer(&f, prec)
    }

    fn div_ball(a: &ArbBall, b: &ArbBall) -> Result<ArbBall> {
        (a.clone() / b.clone()).ok_or_else(|| ValidatedError::DomainViolation {
            what: "division by an interval containing zero".to_string(),
        })
    }

    // ── elementary functions ─────────────────────────────────────────────

    /// `exp(self)`.
    pub fn exp(&self) -> Result<Self> {
        self.check_finite("exp argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "exp argument".into(),
            });
        }
        let e0 = from_float(&m0, self.prec).exp();
        let mut a = Vec::with_capacity(self.order + 1);
        for k in 0..=self.order {
            a.push(Self::div_ball(&e0, &Self::factorial(k, self.prec))?);
        }
        // sup |exp^{(p+1)}| over arg = sup exp over arg.
        let sup = arg.exp();
        let fact = Self::factorial(self.order + 1, self.prec);
        let scale = Self::div_ball(&sup, &fact)?;
        let radius = ub(
            &(scale * ArbBall::from_midpoint_radius(0.0, 0.0, self.prec).clone()
                + ArbBall {
                    mid: delta.delta_pow(&d),
                    rad: Float::new(self.prec),
                    prec: self.prec,
                } * Self::div_ball(&arg.exp(), &fact)?),
        );
        let out = delta.compose(&a, &radius);
        out.check_finite("exp result")?;
        Ok(out)
    }

    /// `log(self)`.  Refuses unless the range is strictly positive.
    pub fn log(&self) -> Result<Self> {
        self.check_finite("log argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let arg_lo = lb(&arg);
        if !strictly_positive(&arg_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "log of an argument whose enclosure reaches 0 or below".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.log().ok_or_else(|| ValidatedError::DomainViolation {
            what: "log expansion point is not positive".into(),
        })?);
        for k in 1..=self.order {
            // aₖ = (-1)^{k-1} / (k · m₀^k)
            let denom = c.powi(k as i64) * ArbBall::from_f64(k as f64, self.prec);
            let mut t = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
            if k % 2 == 0 {
                t = -t;
            }
            a.push(t);
        }
        // |f^{(p+1)}(ξ)| / (p+1)! = 1 / ((p+1)·ξ^{p+1}), maximal at ξ = arg_lo.
        let p1 = self.order + 1;
        let lo_ball = from_float(&arg_lo, self.prec);
        let denom = lo_ball.powi(p1 as i64) * ArbBall::from_f64(p1 as f64, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("log result")?;
        Ok(out)
    }

    /// `1 / self`.  Refuses when the range contains zero.
    pub fn recip(&self) -> Result<Self> {
        self.check_finite("reciprocal argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if contains_zero(&arg) {
            return Err(ValidatedError::DomainViolation {
                what: "reciprocal of an enclosure containing zero".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let mut a = Vec::with_capacity(self.order + 1);
        for k in 0..=self.order {
            // aₖ = (-1)^k / m₀^{k+1}
            let mut t =
                Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &c.powi((k + 1) as i64))?;
            if k % 2 == 1 {
                t = -t;
            }
            a.push(t);
        }
        // |f^{(p+1)}(ξ)| / (p+1)! = 1 / |ξ|^{p+2}, maximal at the mignitude.
        let m = mig(&arg);
        if !strictly_positive(&m) {
            return Err(ValidatedError::DomainViolation {
                what: "reciprocal argument touches zero".into(),
            });
        }
        let denom = from_float(&m, self.prec).powi((self.order + 2) as i64);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("reciprocal result")?;
        Ok(out)
    }

    /// `sqrt(self)`.  Refuses unless the range is strictly positive — at zero
    /// the derivatives of `sqrt` are unbounded, so no Taylor remainder exists.
    pub fn sqrt(&self) -> Result<Self> {
        self.check_finite("sqrt argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let arg_lo = lb(&arg);
        if !strictly_positive(&arg_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "sqrt of an argument whose enclosure reaches 0 or below (derivatives blow up at 0)".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let root = c.sqrt().ok_or_else(|| ValidatedError::DomainViolation {
            what: "sqrt expansion point is negative".into(),
        })?;
        // binom(1/2, k) via bₖ = bₖ₋₁ · (1/2 - (k-1)) / k
        let half = ArbBall::from_f64(0.5, self.prec);
        let mut binom = vec![ArbBall::from_f64(1.0, self.prec)];
        for k in 1..=self.order + 1 {
            let prev = binom[k - 1].clone();
            let num = half.clone() - ArbBall::from_f64((k - 1) as f64, self.prec);
            let t = Self::div_ball(&(prev * num), &ArbBall::from_f64(k as f64, self.prec))?;
            binom.push(t);
        }
        let mut a = Vec::with_capacity(self.order + 1);
        for (k, b) in binom.iter().enumerate().take(self.order + 1) {
            // aₖ = binom(1/2,k) · m₀^{1/2-k}
            let t = Self::div_ball(&(b.clone() * root.clone()), &c.powi(k as i64))?;
            a.push(t);
        }
        // |f^{(p+1)}(ξ)|/(p+1)! = |binom(1/2,p+1)| · ξ^{1/2-(p+1)}, maximal at ξ = arg_lo.
        let p1 = self.order + 1;
        let lo_ball = from_float(&arg_lo, self.prec);
        let lo_root = lo_ball
            .sqrt()
            .ok_or_else(|| ValidatedError::DomainViolation {
                what: "sqrt lower bound is negative".into(),
            })?;
        let scale = Self::div_ball(
            &(symmetric(&mag(&binom[p1]), self.prec) * lo_root),
            &lo_ball.powi(p1 as i64),
        )?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("sqrt result")?;
        Ok(out)
    }

    fn trig(&self, is_sin: bool) -> Result<Self> {
        self.check_finite("trig argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let c = from_float(&m0, self.prec);
        let (s, co) = (c.sin(), c.cos());
        let mut a = Vec::with_capacity(self.order + 1);
        for k in 0..=self.order {
            // dᵏ/dxᵏ sin = sin, cos, -sin, -cos ; cos shifts by one, because
            // cos⁽ᵏ⁾ = sin⁽ᵏ⁺¹⁾. The `(k+1) % 4` phase *is* that shift and
            // already yields the cosine derivative — `k = 0` gives `cos(m₀)`,
            // `k = 1` gives `-sin(m₀)`. A further `-base` for the cosine
            // branch (present until 3.8) negated the whole polynomial while
            // leaving the symmetric remainder bound untouched, so every
            // "validated" cosine came back tight, confident and sign-flipped:
            // `bound_on_box(cos x, x ∈ [1,1])` returned `[-0.54030…, -0.54030…]`
            // for `cos 1 = +0.54030…`, an enclosure that does not contain the
            // value it encloses. Downstream that is a false theorem, not just a
            // wrong number — `verified_no_roots(cos x - 0.9, [0,1])` answered
            // `true` although `arccos(0.9) = 0.451 ∈ [0,1]`. `sin²+cos² = 1` is
            // invariant under the flip, which is why the existing test passed.
            let phase = if is_sin { k % 4 } else { (k + 1) % 4 };
            let base = match phase {
                0 => s.clone(),
                1 => co.clone(),
                2 => -s.clone(),
                _ => -co.clone(),
            };
            a.push(Self::div_ball(&base, &Self::factorial(k, self.prec))?);
        }
        // |sin^{(p+1)}| ≤ 1 and |cos^{(p+1)}| ≤ 1 everywhere.
        let fact = Self::factorial(self.order + 1, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &fact)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("trig result")?;
        Ok(out)
    }

    /// `sin(self)`.
    pub fn sin(&self) -> Result<Self> {
        self.trig(true)
    }

    /// `cos(self)`.
    pub fn cos(&self) -> Result<Self> {
        self.trig(false)
    }

    fn hyp(&self, is_sinh: bool) -> Result<Self> {
        self.check_finite("hyperbolic argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let c = from_float(&m0, self.prec);
        let (sh, ch) = (c.sinh(), c.cosh());
        let mut a = Vec::with_capacity(self.order + 1);
        for k in 0..=self.order {
            let even = k % 2 == 0;
            let base = if is_sinh == even {
                sh.clone()
            } else {
                ch.clone()
            };
            a.push(Self::div_ball(&base, &Self::factorial(k, self.prec))?);
        }
        // Both |sinh| and |cosh| are bounded by cosh(max|x|) on the argument range.
        let m = mag(&arg);
        let sup = from_float(&m, self.prec).cosh();
        let fact = Self::factorial(self.order + 1, self.prec);
        let scale = Self::div_ball(&sup, &fact)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("hyperbolic result")?;
        Ok(out)
    }

    /// `sinh(self)`.
    pub fn sinh(&self) -> Result<Self> {
        self.hyp(true)
    }

    /// `cosh(self)`.
    pub fn cosh(&self) -> Result<Self> {
        self.hyp(false)
    }

    /// `atan(self)`.
    ///
    /// Uses the closed form `f^{(k)}(m₀) = (-1)^{k-1}(k-1)! ρ^{-k} sin(kφ)`
    /// with `ρ = √(1+m₀²)` and `φ = π/2 - atan(m₀)`, which also yields the
    /// clean derivative bound `|f^{(k)}(x)| ≤ (k-1)! (1+x²)^{-k/2}`.
    pub fn atan(&self) -> Result<Self> {
        self.check_finite("atan argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let c = from_float(&m0, self.prec);
        let rho2 = ArbBall::from_f64(1.0, self.prec) + c.clone() * c.clone();
        let rho = rho2.sqrt().ok_or_else(|| ValidatedError::NotFinite {
            what: "atan expansion radius".into(),
        })?;
        let phi = pi_ball(self.prec) * ArbBall::from_f64(0.5, self.prec) - c.atan();
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.atan());
        for k in 1..=self.order {
            let kb = ArbBall::from_f64(k as f64, self.prec);
            let s = (phi.clone() * kb.clone()).sin();
            let denom = rho.powi(k as i64) * kb;
            let mut t = Self::div_ball(&s, &denom)?;
            if k % 2 == 0 {
                t = -t;
            }
            a.push(t);
        }
        // |f^{(p+1)}(ξ)|/(p+1)! ≤ 1 / ((p+1)(1+ξ²)^{(p+1)/2}); worst at min |ξ|.
        let p1 = self.order + 1;
        let m = mig(&arg);
        let mb = from_float(&m, self.prec);
        let base = (ArbBall::from_f64(1.0, self.prec) + mb.clone() * mb)
            .sqrt()
            .ok_or_else(|| ValidatedError::NotFinite {
                what: "atan derivative bound".into(),
            })?;
        let denom = base.powi(p1 as i64) * ArbBall::from_f64(p1 as f64, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("atan result")?;
        Ok(out)
    }

    /// `tan(self) = sin/cos`.  Refuses when `cos` can vanish on the box.
    pub fn tan(&self) -> Result<Self> {
        let c = self.cos()?;
        if contains_zero(&c.range()) {
            return Err(ValidatedError::DomainViolation {
                what: "tan: cos enclosure contains zero (pole in the box)".into(),
            });
        }
        self.sin()?.div(&c)
    }

    /// `tanh(self) = sinh/cosh`.  `cosh ≥ 1`, so this never refuses on domain
    /// grounds.
    pub fn tanh(&self) -> Result<Self> {
        let c = self.cosh()?;
        self.sinh()?.div(&c)
    }

    /// `asin(self) = atan(x / √(1-x²))`.  Refuses when the range reaches
    /// `±1`, where the derivative is unbounded.
    pub fn asin(&self) -> Result<Self> {
        let one = TaylorModel::constant(
            ArbBall::from_f64(1.0, self.prec),
            self.nvars,
            self.order,
            self.prec,
        );
        let inner = one.sub(&self.mul(self));
        if !strictly_positive(&lb(&inner.range())) {
            return Err(ValidatedError::DomainViolation {
                what: "asin: the enclosure of 1-x² reaches 0 or below (|x| ≥ 1)".into(),
            });
        }
        let denom = inner.sqrt()?;
        self.div(&denom)?.atan()
    }

    /// `acos(self) = π/2 - asin(self)`.
    pub fn acos(&self) -> Result<Self> {
        let half_pi = TaylorModel::constant(
            pi_ball(self.prec) * ArbBall::from_f64(0.5, self.prec),
            self.nvars,
            self.order,
            self.prec,
        );
        Ok(half_pi.sub(&self.asin()?))
    }

    /// Upper bound for the series tail `Σ_{k>p} |aₖ|·|δ|ᵏ` of a Taylor
    /// expansion whose coefficients obey `|aₖ| ≤ 1/(k·ρᵏ)` for `k ≥ 1`, where
    /// `ρ` is a lower bound on the distance from the expansion point to the
    /// nearest singularity.  `p1` is `p+1`.
    ///
    /// With `q = |δ|/ρ < 1`, and `1/k ≤ 1/(p+1)` for every `k ≥ p+1`,
    ///
    /// ```text
    /// Σ_{k>p} |aₖ|·|δ|ᵏ  ≤  (1/(p+1))·Σ_{k>p} qᵏ  =  q^{p+1} / [ (p+1)·(1-q) ].
    /// ```
    ///
    /// `t ↦ t^{p+1}/(1-t)` is increasing on `[0,1)`, so rounding `q` **up**
    /// and `1-q` **down** rounds the whole bound up.  `None` when `q ≥ 1`:
    /// outside the disc of convergence the series says nothing, and the
    /// caller must use its Lagrange bound instead.
    ///
    /// This does **not** replace the Lagrange remainder — it is a second,
    /// independent upper bound on the same quantity, and the callers keep
    /// whichever is smaller (a minimum of two valid upper bounds is a valid
    /// upper bound).  Lagrange takes the supremum of `|f⁽ᵖ⁺¹⁾|` over the
    /// *whole* argument enclosure, which for these functions is attained at
    /// the end nearest the singularity and can be three orders of magnitude
    /// above the coefficients at the centre: on `atanh`, `x ∈ [-0.5, 0.5]` at
    /// order 10, Lagrange gives 9.1e-2 and this gives 8.9e-5.
    fn series_tail(delta_mag: &Float, rho_lo: &Float, p1: usize, prec: u32) -> Option<Float> {
        if !strictly_positive(rho_lo) || !delta_mag.is_finite() {
            return None;
        }
        let q = ub(&Self::div_ball(&from_float(delta_mag, prec), &from_float(rho_lo, prec)).ok()?);
        let one_minus_q = lb(&(ArbBall::from_f64(1.0, prec) - from_float(&q, prec)));
        if !strictly_positive(&one_minus_q) {
            return None;
        }
        let num = from_float(&q, prec).powi(p1 as i64);
        let denom = from_float(&one_minus_q, prec) * ArbBall::from_f64(p1 as f64, prec);
        Some(ub(&Self::div_ball(&num, &denom).ok()?))
    }

    /// The smaller of two rigorous upper bounds on the same remainder — still
    /// a rigorous upper bound.  `None` means the second bound did not apply.
    fn tighter(lagrange: Float, tail: Option<Float>) -> Float {
        match tail {
            Some(t) if t < lagrange => t,
            _ => lagrange,
        }
    }

    /// Taylor coefficients `bₙ = w⁽ⁿ⁾(m₀)/n!` for `n = 0..count`, where
    /// `w(x) = (x² + s)^{-1/2}` and `s = ±1`.
    ///
    /// `w` is the derivative of `asinh` (`s = +1`) and of `acosh` (`s = -1`),
    /// so both rules expand with these numbers.
    ///
    /// `w` satisfies `(x² + s)·w′ + x·w = 0`.  Differentiating `n` times with
    /// the Leibniz rule — `x² + s` has only two non-vanishing derivatives,
    /// `2x` and `2` — gives
    ///
    /// ```text
    /// (x² + s)·w⁽ⁿ⁺¹⁾ + (2n+1)·x·w⁽ⁿ⁾ + n²·w⁽ⁿ⁻¹⁾ = 0,
    /// ```
    ///
    /// and dividing through by `n!` turns that into the coefficient
    /// recurrence
    ///
    /// ```text
    /// bₙ₊₁ = -[ (2n+1)·m₀·bₙ + n·bₙ₋₁ ] / [ (n+1)·(m₀² + s) ].
    /// ```
    ///
    /// This is an identity between derivatives **at the single point `m₀`**,
    /// not over an interval, so running it in ball arithmetic is exact up to
    /// the rounding that `ArbBall` already carries: no dependency widening is
    /// possible because there is no interval to widen.
    ///
    /// `m₀² + s` must be strictly positive — the callers establish that from
    /// their own domain guard before calling.
    fn sqrt_recip_coeffs(m0: &ArbBall, s: i32, count: usize, prec: u32) -> Result<Vec<ArbBall>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let q = m0.clone() * m0.clone() + ArbBall::from_f64(f64::from(s), prec);
        let root = q.sqrt().ok_or_else(|| ValidatedError::DomainViolation {
            what: "inverse-hyperbolic expansion point is outside the real domain".into(),
        })?;
        let mut b = Vec::with_capacity(count);
        b.push(Self::div_ball(&ArbBall::from_f64(1.0, prec), &root)?);
        for n in 0..count.saturating_sub(1) {
            let mut num = ArbBall::from_f64((2 * n + 1) as f64, prec) * m0.clone() * b[n].clone();
            if n >= 1 {
                num = num + ArbBall::from_f64(n as f64, prec) * b[n - 1].clone();
            }
            let denom = ArbBall::from_f64((n + 1) as f64, prec) * q.clone();
            b.push(-Self::div_ball(&num, &denom)?);
        }
        Ok(b)
    }

    /// `asinh(self)`.  Real-analytic on all of ℝ, so there is no domain
    /// guard — the only refusals are non-finite enclosures.
    ///
    /// **Coefficients.**  `asinh⁽ᵏ⁾ = w⁽ᵏ⁻¹⁾` for `k ≥ 1` with
    /// `w(x) = (1+x²)^{-1/2}`, so `aₖ = w⁽ᵏ⁻¹⁾(m₀)/k! = b_{k-1}/k` from
    /// `TaylorModel::sqrt_recip_coeffs` with `s = +1`.
    ///
    /// **Remainder.**  On the real axis `w(x) = (x-i)^{-1/2}(x+i)^{-1/2}` for
    /// the branch that is positive there (`x ∓ i` never meets the cut for
    /// real `x`, so a continuous branch exists and both sides are
    /// real-analytic and agree at `x = 0`).  Leibniz on that product gives
    ///
    /// ```text
    /// w⁽ⁿ⁾(x) = (-1)ⁿ Σ_{j=0}^{n} C(n,j)·A_j·A_{n-j}
    ///                  ·(x-i)^{-1/2-j}·(x+i)^{-1/2-(n-j)},
    ///                                            A_j = (2j-1)!!/2ʲ > 0.
    /// ```
    ///
    /// Both `|x-i|` and `|x+i|` equal `ρ = √(1+x²)`, so every term has
    /// modulus `C(n,j)·A_j·A_{n-j}·ρ^{-(n+1)}` and the triangle inequality
    /// gives `|w⁽ⁿ⁾(x)| ≤ ρ^{-(n+1)}·Σ_j C(n,j)A_jA_{n-j}`.  That sum is
    /// exactly `n!`: `Σ_j A_j·tʲ/j! = (1-t)^{-1/2}`, and squaring it gives
    /// `(1-t)^{-1} = Σ_n n!·tⁿ/n!`, whose `n`-th coefficient is the binomial
    /// convolution.  Hence
    ///
    /// ```text
    /// |asinh⁽ᵖ⁺¹⁾(ξ)| / (p+1)!  =  |w⁽ᵖ⁾(ξ)| / (p+1)!
    ///                           ≤  1 / [ (p+1)·(1+ξ²)^{(p+1)/2} ],
    /// ```
    ///
    /// the same expression `atan` uses.  `(1+ξ²)^{-(p+1)/2}` is largest where
    /// `|ξ|` is smallest, so the supremum over the argument enclosure is at
    /// its mignitude.  Nothing here assumes `asinh` (or any derivative of it)
    /// is monotone.
    ///
    /// The rule then keeps the smaller of that and
    /// `TaylorModel::series_tail`'s geometric bound, which is usually far
    /// tighter and is derived from the very same coefficient estimate.
    pub fn asinh(&self) -> Result<Self> {
        self.check_finite("asinh argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "asinh argument".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let b = Self::sqrt_recip_coeffs(&c, 1, self.order, self.prec)?;
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.asinh());
        for k in 1..=self.order {
            a.push(Self::div_ball(
                &b[k - 1],
                &ArbBall::from_f64(k as f64, self.prec),
            )?);
        }
        let p1 = self.order + 1;
        let m = mig(&arg);
        let mb = from_float(&m, self.prec);
        let base = (ArbBall::from_f64(1.0, self.prec) + mb.clone() * mb)
            .sqrt()
            .ok_or_else(|| ValidatedError::NotFinite {
                what: "asinh derivative bound".into(),
            })?;
        let denom = base.powi(p1 as i64) * ArbBall::from_f64(p1 as f64, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
        let lagrange = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        // Second bound: `asinh` is analytic in `|z - m₀| < ρ = √(1+m₀²)` (its
        // branch points are at `±i`, both at distance `ρ` from a real `m₀`),
        // and the same derivative bound gives `|aₖ| ≤ 1/(k·ρᵏ)`.
        let rho = (ArbBall::from_f64(1.0, self.prec) + c.clone() * c.clone())
            .sqrt()
            .ok_or_else(|| ValidatedError::NotFinite {
                what: "asinh convergence radius".into(),
            })?;
        let tail = Self::series_tail(&mag(&d), &lb(&rho), p1, self.prec);
        let radius = Self::tighter(lagrange, tail);
        let out = delta.compose(&a, &radius);
        out.check_finite("asinh result")?;
        Ok(out)
    }

    /// `acosh(self)`.  Refuses unless the argument enclosure lies strictly
    /// above `1`.
    ///
    /// **Domain.**  The principal branch is real only for `x ≥ 1`, and at
    /// `x = 1` the derivative `(x²-1)^{-1/2}` is infinite, so no Taylor
    /// remainder exists there.  The guard is therefore `lb(arg) > 1`,
    /// strictly — an enclosure that merely touches `1` is refused, and so is
    /// one containing a NaN endpoint (the comparison goes through
    /// `strictly_positive`, which answers `false` on NaN).
    ///
    /// **Coefficients.**  `acosh⁽ᵏ⁾ = v⁽ᵏ⁻¹⁾` for `k ≥ 1` with
    /// `v(x) = (x²-1)^{-1/2}`, so `aₖ = b_{k-1}/k` from
    /// `TaylorModel::sqrt_recip_coeffs` with `s = -1`.
    ///
    /// **Remainder.**  For `x > 1` write `v(x) = (x-1)^{-1/2}(x+1)^{-1/2}`
    /// with both factors real and positive.  Leibniz gives
    ///
    /// ```text
    /// v⁽ⁿ⁾(x) = (-1)ⁿ Σ_{j=0}^{n} C(n,j)·A_j·A_{n-j}
    ///                  ·(x-1)^{-1/2-j}·(x+1)^{-1/2-(n-j)},
    ///                                            A_j = (2j-1)!!/2ʲ > 0,
    /// ```
    /// so every term carries the same sign `(-1)ⁿ` and no cancellation can be
    /// exploited — the sum of moduli *is* the modulus of the sum.  Since
    /// `0 < x-1 < x+1` and the exponents are negative,
    /// `(x+1)^{-1/2-(n-j)} ≤ (x-1)^{-1/2-(n-j)}`, and with
    /// `Σ_j C(n,j)A_jA_{n-j} = n!` (the generating-function identity spelled
    /// out on [`TaylorModel::asinh`]) this collapses to
    ///
    /// ```text
    /// |v⁽ⁿ⁾(x)| ≤ n!·(x-1)^{-(n+1)}      for every x > 1,
    /// ```
    ///
    /// hence
    ///
    /// ```text
    /// |acosh⁽ᵖ⁺¹⁾(ξ)| / (p+1)!  =  |v⁽ᵖ⁾(ξ)| / (p+1)!
    ///                           ≤  1 / [ (p+1)·(ξ-1)^{p+1} ].
    /// ```
    ///
    /// The right-hand side is decreasing in `ξ` — a property of the *bound*,
    /// proved above, not an assumed monotonicity of the function — so its
    /// supremum over the argument enclosure sits at the lower endpoint.
    ///
    /// The rule then keeps the smaller of that and
    /// `TaylorModel::series_tail`'s geometric bound.
    pub fn acosh(&self) -> Result<Self> {
        self.check_finite("acosh argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        // `ξ - 1` over the enclosure, rounded **down** by taking the lower
        // bound of the ball rather than subtracting from `lb(arg)` at working
        // precision: a round-to-nearest there could shrink the remainder
        // bound by an ulp, which is exactly the direction a certificate must
        // not move in.
        let excess = lb(&(arg.clone() - ArbBall::from_f64(1.0, self.prec)));
        if !strictly_positive(&excess) {
            return Err(ValidatedError::DomainViolation {
                what: "acosh of an argument whose enclosure reaches 1 or below (the domain is x > 1; the derivative is unbounded at 1)".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let b = Self::sqrt_recip_coeffs(&c, -1, self.order, self.prec)?;
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.acosh().ok_or_else(|| ValidatedError::DomainViolation {
            what: "acosh expansion point is below 1".into(),
        })?);
        for k in 1..=self.order {
            a.push(Self::div_ball(
                &b[k - 1],
                &ArbBall::from_f64(k as f64, self.prec),
            )?);
        }
        let p1 = self.order + 1;
        // `excess` is a rounded-down lower bound on ξ - 1 over the enclosure,
        // so raising it to p+1 and inverting rounds the bound *outward*.
        let denom = from_float(&excess, self.prec).powi(p1 as i64)
            * ArbBall::from_f64(p1 as f64, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &denom)?;
        let lagrange = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        // Second bound: `acosh` is analytic in `|z - m₀| < m₀ - 1` — its
        // branch points are at `±1`, and for `m₀ > 1` the nearer one is `+1`
        // — and the derivative bound above gives `|aₖ| ≤ 1/(k·(m₀-1)ᵏ)`.
        // `m₀ ≥ lb(arg) > 1` because `0 ∈ range(Δ)`, so `m₀ - 1 > 0`.
        let rho = c.clone() - ArbBall::from_f64(1.0, self.prec);
        let tail = Self::series_tail(&mag(&d), &lb(&rho), p1, self.prec);
        let radius = Self::tighter(lagrange, tail);
        let out = delta.compose(&a, &radius);
        out.check_finite("acosh result")?;
        Ok(out)
    }

    /// `atanh(self)`.  Refuses unless the argument enclosure lies strictly
    /// inside `(-1, 1)`.
    ///
    /// **Domain.**  `atanh` is real only on `(-1, 1)` and blows up at both
    /// ends, so the guard is `lb(arg) > -1` **and** `ub(arg) < 1`, both
    /// strict.  Checking only one end would accept `[0.5, 2]`, where the
    /// function is not real at all.
    ///
    /// **Coefficients.**  `atanh′ = g` with
    /// `g(x) = (1-x²)^{-1} = ½[(1-x)^{-1} + (1+x)^{-1}]`, whose derivatives
    /// are elementary:
    ///
    /// ```text
    /// g⁽ⁿ⁾(x) = ½·n!·[ (1-x)^{-(n+1)} + (-1)ⁿ·(1+x)^{-(n+1)} ].
    /// ```
    ///
    /// With `atanh⁽ᵏ⁾ = g⁽ᵏ⁻¹⁾` this gives the closed form
    /// `aₖ = (1/2k)·[ (1-m₀)^{-k} + (-1)^{k-1}·(1+m₀)^{-k} ]` for `k ≥ 1`,
    /// and `a₀ = atanh(m₀)`.  No recurrence is needed.
    ///
    /// **Remainder.**  From the same formula,
    ///
    /// ```text
    /// |atanh⁽ᵖ⁺¹⁾(ξ)| / (p+1)!  =  |g⁽ᵖ⁾(ξ)| / (p+1)!
    ///     ≤ (1 / [2(p+1)])·[ (1-ξ)^{-(p+1)} + (1+ξ)^{-(p+1)} ].
    /// ```
    ///
    /// The two terms are handled separately and each is monotone on the whole
    /// enclosure `[L, U] ⊂ (-1,1)`: `(1-ξ)^{-(p+1)}` increases in `ξ`, so it
    /// is largest at `U`; `(1+ξ)^{-(p+1)}` decreases, so it is largest at
    /// `L`.  Adding the two separate maxima is an upper bound for the sum
    /// even though they are attained at different points, which is why this
    /// needs no monotonicity claim about `g⁽ᵖ⁾` itself.
    ///
    /// That bound is the loosest of the three rules' — it is the supremum of
    /// `|f⁽ᵖ⁺¹⁾|` at the end of the enclosure nearest a pole, which near
    /// `|x| = 1` dwarfs the coefficients at the centre — so the rule keeps
    /// the smaller of it and `TaylorModel::series_tail`'s geometric bound.
    pub fn atanh(&self) -> Result<Self> {
        self.check_finite("atanh argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        // Distance to each pole, rounded **down** in both cases (the lower
        // bound of a ball, not an arithmetic on `lb`/`ub` at working
        // precision, so a round-to-nearest cannot shrink the remainder bound
        // computed from them below).
        let one_b = ArbBall::from_f64(1.0, self.prec);
        let above = lb(&(arg.clone() + one_b.clone()));
        let below = lb(&(one_b - arg.clone()));
        if !strictly_positive(&above) || !strictly_positive(&below) {
            return Err(ValidatedError::DomainViolation {
                what: "atanh of an argument whose enclosure leaves or touches (-1, 1)".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let one = ArbBall::from_f64(1.0, self.prec);
        let minus = one.clone() - c.clone();
        let plus = one.clone() + c.clone();
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.atanh().ok_or_else(|| ValidatedError::DomainViolation {
            what: "atanh expansion point is outside (-1, 1)".into(),
        })?);
        for k in 1..=self.order {
            let pm = Self::div_ball(&one, &minus.powi(k as i64))?;
            let pp = Self::div_ball(&one, &plus.powi(k as i64))?;
            let sum = if k % 2 == 1 { pm + pp } else { pm - pp };
            a.push(Self::div_ball(
                &sum,
                &ArbBall::from_f64(2.0 * k as f64, self.prec),
            )?);
        }
        let p1 = self.order + 1;
        // `below` rounds `1 - U` down and `above` rounds `1 + L` down, so both
        // reciprocal powers below round the derivative bound *up*.
        let hi_term = Self::div_ball(&one, &from_float(&below, self.prec).powi(p1 as i64))?;
        let lo_term = Self::div_ball(&one, &from_float(&above, self.prec).powi(p1 as i64))?;
        let scale = Self::div_ball(
            &(hi_term + lo_term),
            &ArbBall::from_f64(2.0 * p1 as f64, self.prec),
        )?;
        let lagrange = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        // Second bound: `atanh` is analytic in `|z - m₀| < min(1-m₀, 1+m₀)`
        // (poles at `±1`), and its exact coefficients satisfy
        // `|aₖ| ≤ ½[1/(k(1-m₀)ᵏ) + 1/(k(1+m₀)ᵏ)]`, so the tail splits into
        // two `1/(k·ρᵏ)` series, one per pole, each halved.  Both must
        // converge for the sum to be bounded this way; if either does not,
        // the Lagrange bound stands alone.
        let dm = mag(&d);
        let near = Self::series_tail(&dm, &lb(&minus), p1, self.prec);
        let far = Self::series_tail(&dm, &lb(&plus), p1, self.prec);
        let tail = match (near, far) {
            (Some(n), Some(f)) => Some(ub(&Self::div_ball(
                &(from_float(&n, self.prec) + from_float(&f, self.prec)),
                &ArbBall::from_f64(2.0, self.prec),
            )?)),
            _ => None,
        };
        let radius = Self::tighter(lagrange, tail);
        let out = delta.compose(&a, &radius);
        out.check_finite("atanh result")?;
        Ok(out)
    }

    /// Taylor coefficients `bₙ = u⁽ⁿ⁾(m₀)/n!` for `n = 0..count`, where
    /// `u(x) = e^{-x²}` is the Gaussian — the derivative of `erf` up to the
    /// constant `2/√π`.
    ///
    /// `u′ = -2x·u`; differentiating `n` times with Leibniz (`-2x` has one
    /// non-vanishing derivative) gives
    /// `u⁽ⁿ⁺¹⁾ = -2x·u⁽ⁿ⁾ - 2n·u⁽ⁿ⁻¹⁾`, which divided by `n!` is
    ///
    /// ```text
    /// bₙ₊₁ = -2·( m₀·bₙ + bₙ₋₁ ) / (n+1).
    /// ```
    ///
    /// (`u⁽ⁿ⁾(x) = (-1)ⁿ·Hₙ(x)·e^{-x²}` with `Hₙ` the Hermite polynomials;
    /// this is their recurrence in disguise.)  As with
    /// `TaylorModel::sqrt_recip_coeffs` the recurrence holds at the single
    /// point `m₀`, so ball arithmetic runs it without widening.
    fn gaussian_coeffs(m0: &ArbBall, count: usize, prec: u32) -> Result<Vec<ArbBall>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let two = ArbBall::from_f64(2.0, prec);
        let mut b = Vec::with_capacity(count);
        b.push((-(m0.clone() * m0.clone())).exp());
        for n in 0..count.saturating_sub(1) {
            let mut num = m0.clone() * b[n].clone();
            if n >= 1 {
                num = num + b[n - 1].clone();
            }
            let t = Self::div_ball(
                &(two.clone() * num),
                &ArbBall::from_f64((n + 1) as f64, prec),
            )?;
            b.push(-t);
        }
        Ok(b)
    }

    /// Upper bound on `sup_{ξ ∈ arg} |u⁽ⁿ⁾(ξ)| / n!` for the Gaussian
    /// `u(z) = e^{-z²}`, by Cauchy's estimate.
    ///
    /// `u` is entire, so for **every** `r > 0` and every real `a`
    ///
    /// ```text
    /// |u⁽ⁿ⁾(a)| / n!  ≤  max_{|z-a| = r} |u(z)| / rⁿ.
    /// ```
    ///
    /// On that circle `z = a + r·e^{iθ}` with `a` real, writing
    /// `s = cos θ ∈ [-1, 1]`,
    ///
    /// ```text
    /// Re(z²) = a² + 2ar·cos θ + r²·cos 2θ = 2r²·s² + 2ar·s + (a² - r²),
    /// ```
    ///
    /// a parabola in `s` with vertex at `s* = -a/(2r)`.  Its minimum over
    /// `s ∈ [-1,1]` is therefore **exact**, not estimated:
    ///
    /// ```text
    /// min Re(z²) = a²/2 - r²      when |a| ≤ 2r   (vertex inside),
    ///            = (|a| - r)²     when |a| > 2r   (nearest endpoint).
    /// ```
    ///
    /// Both branches are decreasing in `|a|` and agree at `|a| = 2r` (both
    /// give `-r²`), so `max_{|z-a|=r}|u(z)| = exp(-min Re(z²))` is decreasing
    /// in `|a|` and its supremum over the argument enclosure is attained at
    /// the **mignitude** — the point of the enclosure closest to the origin.
    /// No sampling and no monotonicity assumption about `u⁽ⁿ⁾` is involved.
    ///
    /// **The bound holds for every `r > 0`**, so the choice of `r` below is
    /// purely a tightness question and cannot make the result unsound; the
    /// rule evaluates a handful of candidates and keeps the smallest.  A
    /// candidate that overflows is skipped.
    fn gaussian_derivative_bound(arg: &ArbBall, n: usize, prec: u32) -> Result<ArbBall> {
        let t = mig(arg);
        let t_f = t.to_f64();
        if !t_f.is_finite() {
            return Err(ValidatedError::NotFinite {
                what: "erf argument enclosure".into(),
            });
        }
        let nf = n as f64;
        // Near `r = √(n/2)` when the vertex branch applies (`t` small), and
        // near `n/(2t)` when the endpoint branch does (`t` large). Both are
        // hints; every candidate is individually valid.
        let candidates = [
            0.5,
            1.0,
            (nf / 2.0).sqrt().max(0.25),
            if t_f > 0.0 { nf / (2.0 * t_f) } else { 1.0 }.max(0.25),
        ];
        let two = ArbBall::from_f64(2.0, prec);
        let tb = from_float(&t, prec);
        let mut best: Option<ArbBall> = None;
        for r in candidates {
            // `r > 0.0 && r.is_finite()`, spelled so a NaN candidate is
            // skipped rather than silently compared.
            if !matches!(r.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) || !r.is_finite() {
                continue;
            }
            let rb = ArbBall::from_f64(r, prec);
            // `exp(-min Re)`. The vertex value `a²/2 - r²` is the minimum of
            // the parabola over *all* `s`, so it is a lower bound for the
            // constrained minimum whichever branch is really in force —
            // taking it always yields a sound (merely looser) bound. The
            // endpoint branch is therefore only taken with a margin clear of
            // `|a| = 2r`, so that an `f64` rounding in this comparison can
            // never select it where it does not apply.
            let expo = if t_f <= 2.5 * r {
                rb.clone() * rb.clone() - Self::div_ball(&(tb.clone() * tb.clone()), &two)?
            } else {
                -((tb.clone() - rb.clone()) * (tb.clone() - rb.clone()))
            };
            let cand = match Self::div_ball(&expo.exp(), &rb.powi(n as i64)) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if !is_finite(&cand) {
                continue;
            }
            let keep = match &best {
                Some(b) => ub(&cand) < ub(b),
                None => true,
            };
            if keep {
                best = Some(cand);
            }
        }
        best.ok_or_else(|| ValidatedError::NotFinite {
            what: "erf derivative bound".into(),
        })
    }

    /// `2/√π`, the constant in `erf′`.
    fn two_over_sqrt_pi(prec: u32) -> Result<ArbBall> {
        let root = pi_ball(prec)
            .sqrt()
            .ok_or_else(|| ValidatedError::NotFinite {
                what: "√π".into()
            })?;
        Self::div_ball(&ArbBall::from_f64(2.0, prec), &root)
    }

    /// `erf(self)`.  Entire on ℝ, so there is no domain guard.
    ///
    /// **Coefficients.**  `erf′ = (2/√π)·u` with `u(x) = e^{-x²}`, so
    /// `erf⁽ᵏ⁾ = (2/√π)·u⁽ᵏ⁻¹⁾` for `k ≥ 1` and
    /// `aₖ = (2/√π)·b_{k-1}/k` from `TaylorModel::gaussian_coeffs`.
    ///
    /// **Remainder.**
    ///
    /// ```text
    /// |erf⁽ᵖ⁺¹⁾(ξ)| / (p+1)!  =  (2/√π)·|u⁽ᵖ⁾(ξ)| / (p+1)!
    ///                         =  (2/√π)·[ |u⁽ᵖ⁾(ξ)|/p! ] / (p+1),
    /// ```
    ///
    /// and the bracket is bounded over the whole argument enclosure by
    /// `TaylorModel::gaussian_derivative_bound` — a Cauchy estimate, which
    /// needs only that `e^{-z²}` is entire.  No property of `erf` itself
    /// (monotonicity, boundedness by 1, the asymptotic expansion) is used,
    /// so there is nothing here that a wide or oddly-placed box can
    /// invalidate.
    pub fn erf(&self) -> Result<Self> {
        self.check_finite("erf argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "erf argument".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let k2 = Self::two_over_sqrt_pi(self.prec)?;
        let b = Self::gaussian_coeffs(&c, self.order, self.prec)?;
        let mut a = Vec::with_capacity(self.order + 1);
        a.push(c.erf());
        for k in 1..=self.order {
            a.push(Self::div_ball(
                &(k2.clone() * b[k - 1].clone()),
                &ArbBall::from_f64(k as f64, self.prec),
            )?);
        }
        let p1 = self.order + 1;
        let dbound = Self::gaussian_derivative_bound(&arg, self.order, self.prec)?;
        let scale = Self::div_ball(&(k2 * dbound), &ArbBall::from_f64(p1 as f64, self.prec))?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("erf result")?;
        Ok(out)
    }

    /// `erfc(self) = 1 - erf(self)`.
    ///
    /// The identity is exact, and negating a Taylor model negates its
    /// remainder interval without widening it, so this inherits
    /// [`TaylorModel::erf`]'s rigour unchanged.  It is *not* evaluated
    /// through the asymptotic continued fraction that a floating-point
    /// `erfc` uses for large `x`: this rule loses relative accuracy in the
    /// far tail exactly as the subtraction says it does, which is honest
    /// rather than tight.
    pub fn erfc(&self) -> Result<Self> {
        let one = TaylorModel::constant(
            ArbBall::from_f64(1.0, self.prec),
            self.nvars,
            self.order,
            self.prec,
        );
        Ok(one.sub(&self.erf()?))
    }

    /// `self^e` for a real constant exponent, via `exp(e · log(self))`.
    /// Requires a strictly positive base.
    pub fn pow_const(&self, e: &ArbBall) -> Result<Self> {
        let l = self.log()?;
        l.scale(e).exp()
    }

    /// `|self|`.  Only defined when the range has a determined sign — `abs` is
    /// not differentiable at zero, so a straddling box is refused.
    pub fn abs(&self) -> Result<Self> {
        let r = self.range();
        if lb(&r) >= 0 {
            Ok(self.clone())
        } else if ub(&r) <= 0 {
            Ok(self.neg())
        } else {
            Err(ValidatedError::DomainViolation {
                what: "abs over a box whose enclosure straddles zero is not smooth".into(),
            })
        }
    }

    // ── integration (1-D) ────────────────────────────────────────────────

    /// Rigorous enclosure of `∫_{-1}^{1} f(u) du` in normalised coordinates,
    /// for a **univariate** model (`nvars() == 1`).
    ///
    /// The polynomial part integrates termwise exactly:
    /// `∫_{-1}^{1} u^k du = 2/(k+1)` for even `k` and `0` for odd `k` (an odd
    /// monomial is an odd function on the symmetric domain, so it integrates
    /// to exactly zero — this is where Taylor models beat plain
    /// range-times-width quadrature: the sign cancellation is exact, not
    /// approximated). The remainder `I` bounds `f - P` uniformly over
    /// `[-1,1]`, so `∫_{-1}^{1} (f - P) du ∈ 2·I`.
    ///
    /// Used by [`crate::validated::bounds::verified_integral`] to turn a
    /// local Taylor model on a sub-interval into a rigorous enclosure of the
    /// local piece of a definite integral (after scaling by the sub-interval
    /// radius to undo the `x = c + r·u` change of variables).
    ///
    /// Returns [`ValidatedError::InvalidInput`] if `nvars() != 1` — this
    /// method is not a multivariate quadrature rule.
    pub fn integrate_normalized_1d(&self) -> Result<ArbBall> {
        if self.nvars != 1 {
            return Err(ValidatedError::InvalidInput {
                what: "integrate_normalized_1d requires a univariate Taylor model".into(),
            });
        }
        let two = ArbBall::from_f64(2.0, self.prec);
        let mut acc = ArbBall::from_f64(0.0, self.prec);
        for (idx, c) in &self.coeffs {
            let k = idx[0];
            if k % 2 == 0 {
                let denom = ArbBall::from_f64((k + 1) as f64, self.prec);
                let term = Self::div_ball(&(two.clone() * c.clone()), &denom)?;
                acc = acc + term;
            }
            // Odd-degree monomials integrate to exactly zero over [-1,1].
        }
        Ok(acc + two * self.remainder.clone())
    }
}

// ---------------------------------------------------------------------------
// Expression → Taylor model
// ---------------------------------------------------------------------------

/// Evaluation context binding box variables to normalised coordinates.
pub struct TaylorContext<'a> {
    pool: &'a ExprPool,
    vars: Vec<ExprId>,
    centers: Vec<Float>,
    radii: Vec<Float>,
    order: usize,
    prec: u32,
    memo: HashMap<ExprId, TaylorModel>,
}

impl<'a> TaylorContext<'a> {
    /// Build a context over the box `[lo_i, hi_i]` for each listed variable.
    pub fn new(
        pool: &'a ExprPool,
        boxes: &[(ExprId, Float, Float)],
        order: usize,
        prec: u32,
    ) -> Result<Self> {
        if boxes.is_empty() {
            return Err(ValidatedError::InvalidInput {
                what: "the box must constrain at least one variable".into(),
            });
        }
        if order == 0 || order > MAX_ORDER {
            return Err(ValidatedError::InvalidInput {
                what: format!("Taylor order must be in 1..={MAX_ORDER}"),
            });
        }
        let mut vars = Vec::new();
        let mut centers = Vec::new();
        let mut radii = Vec::new();
        for (v, lo, hi) in boxes {
            if !(lo.is_finite() && hi.is_finite()) {
                return Err(ValidatedError::InvalidInput {
                    what: "box endpoints must be finite (improper domains are not supported)"
                        .into(),
                });
            }
            if lo > hi {
                return Err(ValidatedError::InvalidInput {
                    what: "box interval has lo > hi".into(),
                });
            }
            let b = from_bounds(lo, hi, prec);
            vars.push(*v);
            centers.push(b.mid.clone());
            radii.push(b.rad.clone());
        }
        Ok(TaylorContext {
            pool,
            vars,
            centers,
            radii,
            order,
            prec,
            memo: HashMap::new(),
        })
    }

    /// Truncation order.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Working precision.
    pub fn prec(&self) -> u32 {
        self.prec
    }

    fn nvars(&self) -> usize {
        self.vars.len()
    }

    fn konst(&self, b: ArbBall) -> TaylorModel {
        TaylorModel::constant(b, self.nvars(), self.order, self.prec)
    }

    /// Build a Taylor model for `expr` over this box.
    pub fn eval(&mut self, expr: ExprId) -> Result<TaylorModel> {
        if let Some(m) = self.memo.get(&expr) {
            return Ok(m.clone());
        }
        let m = self.eval_uncached(expr)?;
        self.memo.insert(expr, m.clone());
        Ok(m)
    }

    fn eval_uncached(&mut self, expr: ExprId) -> Result<TaylorModel> {
        match self.pool.get(expr) {
            ExprData::Integer(n) => Ok(self.konst(ArbBall::from_integer(&n.0, self.prec))),
            ExprData::Rational(r) => Ok(self.konst(ArbBall::from_rational(&r.0, self.prec))),
            ExprData::Float(f) => Ok(self.konst(from_float(
                &Float::with_val(self.prec, f.inner.to_f64()),
                self.prec,
            ))),
            ExprData::Symbol { name, .. } => {
                if let Some(i) = self.vars.iter().position(|&v| v == expr) {
                    Ok(TaylorModel::variable(
                        i,
                        &self.centers[i].clone(),
                        &self.radii[i].clone(),
                        self.nvars(),
                        self.order,
                        self.prec,
                    ))
                } else {
                    Err(ValidatedError::UnboundSymbol {
                        name: name.to_string(),
                    })
                }
            }
            ExprData::Add(args) => {
                let mut acc = TaylorModel::zero(self.nvars(), self.order, self.prec);
                for a in args {
                    let t = self.eval(a)?;
                    acc = acc.add(&t);
                }
                Ok(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = self.konst(ArbBall::from_f64(1.0, self.prec));
                for a in args {
                    let t = self.eval(a)?;
                    acc = acc.mul(&t);
                }
                Ok(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = self.eval(base)?;
                match self.pool.get(exp) {
                    ExprData::Integer(n) => {
                        let nv = n.0.to_i64().ok_or_else(|| ValidatedError::Unsupported {
                            what: "integer exponent does not fit in i64".into(),
                        })?;
                        b.powi(nv)
                    }
                    ExprData::Rational(r) => b.pow_const(&ArbBall::from_rational(&r.0, self.prec)),
                    ExprData::Float(f) => b.pow_const(&from_float(
                        &Float::with_val(self.prec, f.inner.to_f64()),
                        self.prec,
                    )),
                    _ => {
                        // Symbolic exponent: only sound via exp(e·log b), which
                        // needs a strictly positive base.
                        let e = self.eval(exp)?;
                        let l = b.log()?;
                        l.mul(&e).exp()
                    }
                }
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let x = self.eval(args[0])?;
                match name.as_str() {
                    "exp" => x.exp(),
                    "log" | "ln" => x.log(),
                    "sqrt" => x.sqrt(),
                    "sin" => x.sin(),
                    "cos" => x.cos(),
                    "tan" => x.tan(),
                    "asin" => x.asin(),
                    "acos" => x.acos(),
                    "atan" => x.atan(),
                    "sinh" => x.sinh(),
                    "cosh" => x.cosh(),
                    "tanh" => x.tanh(),
                    "asinh" => x.asinh(),
                    "acosh" => x.acosh(),
                    "atanh" => x.atanh(),
                    "erf" => x.erf(),
                    "erfc" => x.erfc(),
                    "abs" => x.abs(),
                    other => Err(ValidatedError::Unsupported {
                        what: format!("function `{other}`"),
                    }),
                }
            }
            ExprData::Func { name, args } => Err(ValidatedError::Unsupported {
                what: format!("function `{name}` with {} arguments", args.len()),
            }),
            other => Err(ValidatedError::Unsupported {
                what: format!("expression node {other:?}"),
            }),
        }
    }
}

/// One-shot Taylor model range enclosure for `expr` over `boxes`, without any
/// subdivision.  Mostly useful for comparing against plain interval
/// evaluation; [`super::bounds::bound_on_box`] is the practical entry point.
pub fn taylor_range(
    expr: ExprId,
    pool: &ExprPool,
    boxes: &[(ExprId, Float, Float)],
    order: usize,
    prec: u32,
) -> Result<ArbBall> {
    let mut ctx = TaylorContext::new(pool, boxes, order, prec)?;
    let tm = ctx.eval(expr)?;
    let r = tm.range();
    if !is_finite(&r) {
        return Err(ValidatedError::NotFinite {
            what: "range enclosure".into(),
        });
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `a - b` (the pool has no `sub`; subtraction is `a + (-1)·b`).
    fn sub(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), b])])
    }

    /// `a / b` (the pool has no `div`; division is `a · b^(-1)`).
    fn div(pool: &ExprPool, a: ExprId, b: ExprId) -> ExprId {
        pool.mul(vec![a, pool.pow(b, pool.integer(-1_i32))])
    }
    use crate::kernel::Domain;

    const P: u32 = 128;

    fn f(v: f64) -> Float {
        Float::with_val(P, v)
    }

    fn range_of(expr: ExprId, pool: &ExprPool, b: &[(ExprId, f64, f64)], order: usize) -> ArbBall {
        let boxes: Vec<_> = b.iter().map(|(v, lo, hi)| (*v, f(*lo), f(*hi))).collect();
        taylor_range(expr, pool, &boxes, order, P).unwrap()
    }

    /// A validated enclosure that does not contain the value it encloses is the
    /// worst failure this module can have, and `sin²+cos²=1` cannot see a sign
    /// flip. Pin the *value*, at a degenerate box, against the hand constant.
    #[test]
    fn cos_enclosure_contains_cos_of_the_point() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for point in [1.0_f64, 0.0, 2.5, -1.25, 3.5] {
            let expected = point.cos();
            let r = range_of(pool.func("cos", vec![x]), &pool, &[(x, point, point)], 6);
            // A degenerate box gives a tight ball; the tolerance only absorbs
            // the last f64 ulp, and a sign flip misses it by ~2·|cos(point)|.
            assert!(
                (r.mid_f64() - expected).abs() < 1e-12 && r.rad_f64() < 1e-12,
                "cos({point}) = {expected} but the enclosure is {r:?}"
            );
        }
        // …and over a genuine box: cos is ≥ cos(1) > 0 on [0, 1].
        let r = range_of(pool.func("cos", vec![x]), &pool, &[(x, 0.0, 1.0)], 8);
        assert!(r.lo() > 0.0, "cos > 0 on [0,1] but enclosure is {r:?}");
        assert!(r.hi() >= 1.0, "cos(0) = 1 must be enclosed, got {r:?}");
    }

    #[test]
    fn dependency_cancellation_x_minus_x() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = sub(&pool, x, x);
        let r = range_of(e, &pool, &[(x, -1.0, 1.0)], 4);
        // Taylor models cancel symbolically: the enclosure must be ~{0}.
        assert!(r.rad_f64() < 1e-20, "rad = {}", r.rad_f64());
        assert!(r.contains(0.0));
    }

    #[test]
    fn dependency_x_times_one_minus_x() {
        // x(1-x) on [0,1] has true range [0, 1/4].  Naive interval arithmetic
        // gives [0,1]; the degree-2 Taylor model must be exact modulo rounding.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let e = pool.mul(vec![x, sub(&pool, one, x)]);
        let r = range_of(e, &pool, &[(x, 0.0, 1.0)], 4);
        assert!(r.lo() <= 0.0);
        assert!(r.hi() >= 0.25);
        // The u² term contributes [0,1]·(-1/4) so the bound is [-1/4, 1/4]+1/4.
        assert!(r.hi() < 0.30, "hi = {}", r.hi().to_f64());
    }

    #[test]
    fn taylor_beats_interval_on_polynomial() {
        use crate::ball::IntervalEval;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        // (x-1)^4, x ∈ [0.5, 1.5]:  true range [0, 1/16].
        let e = pool.pow(sub(&pool, x, pool.integer(1_i32)), pool.integer(4_i32));
        let tm = range_of(e, &pool, &[(x, 0.5, 1.5)], 6);

        let mut ev = IntervalEval::new(P);
        ev.bind(x, ArbBall::from_midpoint_radius(1.0, 0.5, P));
        let iv = ev.eval(e, &pool).unwrap();

        assert!(tm.lo() <= 0.0 && tm.hi() >= 0.0625);
        assert!(
            width_of(&tm) <= width_of(&iv),
            "taylor {} should not be wider than interval {}",
            width_of(&tm),
            width_of(&iv)
        );
    }

    fn width_of(b: &ArbBall) -> f64 {
        b.rad_f64() * 2.0
    }

    #[test]
    fn exp_encloses_dense_samples() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("exp", vec![x]);
        let r = range_of(e, &pool, &[(x, -1.0, 2.0)], 8);
        for i in 0..=300 {
            let t = -1.0 + 3.0 * (i as f64) / 300.0;
            assert!(r.contains(t.exp()), "exp({t}) escaped {r}");
        }
    }

    #[test]
    fn sin_encloses_dense_samples() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("sin", vec![x]);
        let r = range_of(e, &pool, &[(x, 0.0, 3.0)], 8);
        for i in 0..=300 {
            let t = 3.0 * (i as f64) / 300.0;
            assert!(r.contains(t.sin()), "sin({t}) escaped {r}");
        }
    }

    #[test]
    fn log_sqrt_atan_enclose_dense_samples() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (name, fun) in [
            ("log", f64::ln as fn(f64) -> f64),
            ("sqrt", f64::sqrt),
            ("atan", f64::atan),
        ] {
            let e = pool.func(name, vec![x]);
            let r = range_of(e, &pool, &[(x, 0.5, 2.0)], 8);
            for i in 0..=200 {
                let t = 0.5 + 1.5 * (i as f64) / 200.0;
                assert!(r.contains(fun(t)), "{name}({t}) escaped {r}");
            }
        }
    }

    #[test]
    fn tanh_and_hyperbolics_enclose_samples() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (name, fun) in [
            ("sinh", f64::sinh as fn(f64) -> f64),
            ("cosh", f64::cosh),
            ("tanh", f64::tanh),
        ] {
            let e = pool.func(name, vec![x]);
            let r = range_of(e, &pool, &[(x, -1.0, 1.0)], 8);
            for i in 0..=200 {
                let t = -1.0 + 2.0 * (i as f64) / 200.0;
                assert!(r.contains(fun(t)), "{name}({t}) escaped {r}");
            }
        }
    }

    #[test]
    fn reciprocal_encloses_samples_and_refuses_across_pole() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = div(&pool, pool.integer(1_i32), x);
        let r = range_of(e, &pool, &[(x, 1.0, 3.0)], 8);
        for i in 0..=200 {
            let t = 1.0 + 2.0 * (i as f64) / 200.0;
            assert!(r.contains(1.0 / t), "1/{t} escaped {r}");
        }

        let boxes = vec![(x, f(-1.0), f(1.0))];
        let err = taylor_range(e, &pool, &boxes, 8, P).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
    }

    #[test]
    fn log_refuses_when_box_reaches_zero() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("log", vec![x]);
        let boxes = vec![(x, f(0.0), f(1.0))];
        let err = taylor_range(e, &pool, &boxes, 6, P).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-003");
    }

    // ── inverse hyperbolics ──────────────────────────────────────────────
    //
    // The failure mode these guard against is the one the `cos` sign flip and
    // the `bessel_jn` endpoint hull both had: an enclosure that is returned,
    // is tight, and does *not* contain the value it claims to enclose. So
    // every test below checks containment against densely sampled true
    // values, and every off-domain box is checked to produce a refusal rather
    // than a number.

    fn tm_range(name: &str, lo: f64, hi: f64, order: usize) -> Result<ArbBall> {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func(name, vec![x]);
        taylor_range(e, &pool, &[(x, f(lo), f(hi))], order, P)
    }

    /// The true value at `t`, computed at **higher precision than the
    /// enclosure**.
    ///
    /// An `f64` library call is accurate to ~1e-16; these enclosures are
    /// routinely tight to 1e-22 (`asinh` on `[-100,-99]` at order 8 has a
    /// remainder of 2.4e-22, and the polynomial bound is *attained* at the
    /// endpoint). Checking containment against `f64` therefore reports a
    /// perfectly correct enclosure as broken — and, in the other direction,
    /// would drown a real 1e-17 error. `P + 64` bits is ~1e-57.
    fn ref_val(name: &str, t: &Float) -> Float {
        let v = Float::with_val(P + 64, t);
        match name {
            "asinh" => v.asinh(),
            "acosh" => v.acosh(),
            "atanh" => v.atanh(),
            "erf" => v.erf(),
            "erfc" => v.erfc(),
            _ => unreachable!("no reference for `{name}`"),
        }
    }

    /// `n + 1` sample points spanning `[lo, hi]`, computed at high precision
    /// and clamped into the box.
    ///
    /// Clamping matters: `lo + (hi-lo)·k/n` can round a hair past `hi`, and a
    /// point outside the box is not something the enclosure ever promised to
    /// cover — that would be a test bug reported as a soundness failure.
    fn samples(lo: f64, hi: f64, n: usize) -> Vec<Float> {
        let flo = Float::with_val(P + 64, lo);
        let fhi = Float::with_val(P + 64, hi);
        (0..=n)
            .map(|k| {
                let frac = Float::with_val(P + 64, k as f64) / Float::with_val(P + 64, n as f64);
                let mut t = flo.clone() + (fhi.clone() - flo.clone()) * frac;
                if t < flo {
                    t = flo.clone();
                }
                if t > fhi {
                    t = fhi.clone();
                }
                t
            })
            .collect()
    }

    /// Assert that `r` really brackets `name(t)`.
    fn assert_encloses(name: &str, r: &ArbBall, t: &Float, ctx: &str) {
        let truth = ref_val(name, t);
        assert!(
            r.lo() <= truth && truth <= r.hi(),
            "{name}({t}) = {truth} escaped [{}, {}] {ctx}",
            r.lo(),
            r.hi()
        );
    }

    /// Every sample of the true function on the box is inside the enclosure,
    /// on boxes that run right up to the domain boundary.
    #[test]
    fn inverse_hyperbolics_enclose_dense_samples() {
        let cases: [(&str, &[(f64, f64)]); 3] = [
            (
                "asinh",
                &[
                    (-1.0, 1.0),
                    (0.0, 3.0),
                    (-5.0, -4.0),
                    (2.0, 2.5),
                    (-0.25, 0.25),
                    (7.0, 9.0),
                    (-100.0, -99.0),
                ],
            ),
            (
                "acosh",
                &[
                    (1.5, 2.0),
                    (1.05, 1.1),
                    (1.001, 1.0011),
                    (3.0, 8.0),
                    (10.0, 10.5),
                ],
            ),
            (
                "atanh",
                &[
                    (-0.5, 0.5),
                    (0.9, 0.95),
                    (-0.999, -0.998),
                    (0.0, 0.25),
                    (-0.3, 0.7),
                    (0.999_999, 0.999_999_5),
                ],
            ),
        ];
        for (name, boxes) in cases {
            for &(lo, hi) in boxes {
                let r = tm_range(name, lo, hi, 8)
                    .unwrap_or_else(|e| panic!("{name} on [{lo},{hi}]: {e}"));
                for t in samples(lo, hi, 200) {
                    assert_encloses(name, &r, &t, &format!("on [{lo},{hi}]"));
                }
            }
        }
    }

    /// A degenerate box is a point evaluation: pin the *value*, which a sign
    /// flip or a wrong expansion point cannot survive even though a
    /// containment sweep over a wide box sometimes can.
    #[test]
    fn inverse_hyperbolic_point_values_are_pinned() {
        let cases: [(&str, &[f64]); 3] = [
            ("asinh", &[0.0, 1.0, -1.0, 2.75, -3.5, -600.0]),
            ("acosh", &[1.25, 2.0, 5.5, 40.0]),
            ("atanh", &[0.0, 0.5, -0.5, 0.99, -0.875]),
        ];
        for (name, points) in cases {
            for &p in points {
                let r = tm_range(name, p, p, 6).unwrap();
                let t = Float::with_val(P, p);
                assert_encloses(name, &r, &t, "at a degenerate box");
                assert!(
                    r.rad_f64() < 1e-25,
                    "{name}({p}) should be a point evaluation, got radius {}",
                    r.rad_f64()
                );
            }
        }
    }

    /// The enclosures have to be *useful*, not merely true: a rule returning
    /// `[-∞, ∞]` would pass every containment test above.
    ///
    /// The yardstick is the width of the true range. A single un-subdivided
    /// Taylor model bounds each monomial `uᵏ` over `[-1,1]` independently, so
    /// some overshoot is structural (`bound_on_box` removes it by
    /// subdividing); a factor of two says the polynomial part is doing real
    /// work and the remainder is not dominating.
    #[test]
    fn inverse_hyperbolic_enclosures_are_tight() {
        for (name, lo, hi) in [
            ("asinh", 0.5, 1.5),
            ("asinh", -3.0, -2.0),
            ("acosh", 1.5, 2.5),
            ("acosh", 1.1, 1.2),
            ("atanh", -0.4, 0.4),
            ("atanh", 0.8, 0.9),
        ] {
            let r = tm_range(name, lo, hi, 10).unwrap();
            let span = ref_val(name, &Float::with_val(P + 64, hi))
                - ref_val(name, &Float::with_val(P + 64, lo));
            let span = span.abs().to_f64();
            let got = width_of(&r);
            assert!(
                got <= 2.0 * span,
                "{name} on [{lo},{hi}]: enclosure width {got} against a true range of {span}"
            );
        }
    }

    /// Off-domain and boundary-touching boxes must refuse, not answer.
    ///
    /// `E-VALIDATED-003` is a *domain* violation. `E-VALIDATED-001` would be
    /// wrong here in an interesting way: it would say the rule does not
    /// exist, sending a caller off a route that works one box over.
    #[test]
    fn inverse_hyperbolics_refuse_off_domain_boxes() {
        for (name, lo, hi) in [
            // acosh: the domain is x > 1, and the derivative is unbounded at 1.
            ("acosh", 0.0, 0.5),
            ("acosh", -2.0, -1.0),
            ("acosh", 0.5, 2.0),
            ("acosh", 1.0, 2.0), // touches the branch point
            ("acosh", 0.999_999, 1.5),
            ("acosh", -1.0, 1.0),
            // atanh: the domain is |x| < 1, and *both* ends have to be checked
            // — a guard on one end alone accepts [0.5, 2], where the function
            // is not real at all.
            ("atanh", 1.5, 2.0),
            ("atanh", -2.0, -1.5),
            ("atanh", 0.5, 2.0),
            ("atanh", -2.0, 0.5),
            ("atanh", -1.0, 0.5), // touches -1
            ("atanh", -0.5, 1.0), // touches +1
            ("atanh", -2.0, 2.0),
        ] {
            match tm_range(name, lo, hi, 6) {
                Ok(r) => panic!("{name} on [{lo},{hi}] is off-domain but returned {r}"),
                Err(e) => assert_eq!(
                    crate::errors::AlkahestError::code(&e),
                    "E-VALIDATED-003",
                    "{name} on [{lo},{hi}] refused with the wrong error: {e}"
                ),
            }
        }
    }

    /// `asinh` is entire, so no box refuses on domain grounds — including
    /// boxes straddling zero and boxes far out on the negative axis, where an
    /// implementation routed through `log(x + √(1+x²))` would lose the
    /// argument to cancellation and refuse.
    #[test]
    fn asinh_never_refuses_on_domain_grounds() {
        for (lo, hi) in [
            (-1.0, 1.0),
            (-100.0, -99.0),
            (-30.0, 30.0),
            (0.0, 0.0),
            (-1e6, -1e6),
            (-1e6, -999_999.0),
        ] {
            let r = tm_range("asinh", lo, hi, 6).unwrap_or_else(|e| panic!("[{lo},{hi}]: {e}"));
            for t in samples(lo, hi, 20) {
                assert_encloses("asinh", &r, &t, &format!("on [{lo},{hi}]"));
            }
        }
    }

    /// Randomised sweep: 200 boxes per function drawn across the domain, each
    /// checked for containment of densely sampled true values.
    ///
    /// A seeded LCG rather than a proptest strategy, because what matters is
    /// the *breadth* of box shapes hit (centre, width and distance to the
    /// boundary all varying together) and a fixed seed reproduces exactly.
    #[test]
    fn inverse_hyperbolics_random_box_sweep() {
        let mut state: u64 = 0x0005_deec_e66d;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for name in ["asinh", "acosh", "atanh"] {
            let mut checked = 0usize;
            for _ in 0..200 {
                let (lo, hi) = match name {
                    // asinh: anywhere on ℝ, widths from tiny to wide.
                    "asinh" => {
                        let c = (next() - 0.5) * 20.0;
                        let w = next().powi(3) * 4.0;
                        (c - w, c + w)
                    }
                    // acosh: 1 + a positive offset, so the branch point is
                    // approached from inside but never reached.
                    "acosh" => {
                        let lo = 1.0 + next().powi(4) * 30.0 + 1e-3;
                        let w = next().powi(3) * 2.0;
                        (lo, lo + w)
                    }
                    // atanh: strictly inside (-1, 1) by construction, with the
                    // width scaled by the distance to whichever end is nearer.
                    _ => {
                        let c = (next() - 0.5) * 1.9;
                        let slack = (1.0 - c.abs()) * 0.9;
                        let w = next().powi(3) * slack;
                        (c - w, c + w)
                    }
                };
                let r = match tm_range(name, lo, hi, 8) {
                    Ok(r) => r,
                    // A refusal is always sound — only a *bound* can be wrong,
                    // and `inverse_hyperbolics_refuse_off_domain_boxes` pins
                    // that refusals happen for the right reason.
                    Err(_) => continue,
                };
                checked += 1;
                for t in samples(lo, hi, 40) {
                    assert_encloses(name, &r, &t, &format!("on [{lo}, {hi}]"));
                }
            }
            assert!(
                checked > 100,
                "{name}: only {checked}/200 boxes produced a bound — the sweep is not exercising the rule"
            );
        }
    }

    /// The identities that tie the new rules to the existing ones. On their
    /// own these cannot catch a sign flip (`sinh(asinh x) = x` is the same
    /// shape of test as `sin²+cos² = 1`, which is exactly what missed the
    /// 3.8 cosine flip), which is why they sit *next to* the containment
    /// tests rather than instead of them.
    #[test]
    fn inverse_hyperbolics_agree_with_their_inverses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (outer, inner, lo, hi, slack) in [
            ("sinh", "asinh", 0.25, 0.75, 0.05),
            ("tanh", "atanh", -0.5, 0.5, 0.1),
            ("cosh", "acosh", 1.5, 2.5, 0.1),
        ] {
            let e = pool.func(outer, vec![pool.func(inner, vec![x])]);
            let r = range_of(e, &pool, &[(x, lo, hi)], 10);
            assert!(
                r.lo() <= lo && r.hi() >= hi,
                "{outer}({inner}(x)) must enclose [{lo},{hi}], got {r:?}"
            );
            assert!(
                r.lo() > lo - slack && r.hi() < hi + slack,
                "{outer}({inner}(x)) = x is loose: {r:?}"
            );
        }
    }

    // ── erf / erfc ───────────────────────────────────────────────────────

    /// Containment for `erf`/`erfc` across the transition region and out into
    /// both tails, where the coefficient recurrence alternates sign and could
    /// plausibly lose the value.
    #[test]
    fn erf_family_encloses_dense_samples() {
        for name in ["erf", "erfc"] {
            for &(lo, hi) in &[
                (-1.0_f64, 1.0_f64),
                (0.0, 0.5),
                (-3.0, -2.5),
                (1.5, 2.0),
                (2.0, 6.0),
                (-0.125, 0.125),
                (-8.0, -7.5),
            ] {
                let r = tm_range(name, lo, hi, 8)
                    .unwrap_or_else(|e| panic!("{name} on [{lo},{hi}]: {e}"));
                for t in samples(lo, hi, 200) {
                    assert_encloses(name, &r, &t, &format!("on [{lo},{hi}]"));
                }
            }
        }
    }

    /// `erf` is entire — no box may refuse on domain grounds.
    #[test]
    fn erf_never_refuses_on_domain_grounds() {
        for (lo, hi) in [(-1.0, 1.0), (-50.0, 50.0), (0.0, 0.0), (20.0, 21.0)] {
            for name in ["erf", "erfc"] {
                tm_range(name, lo, hi, 6)
                    .unwrap_or_else(|e| panic!("{name} on [{lo},{hi}] refused: {e}"));
            }
        }
    }

    /// `erf(x) + erfc(x) = 1` exactly, and — the part the identity cannot
    /// see — each half separately brackets its own value at a point.
    #[test]
    fn erf_and_erfc_are_complementary_and_individually_pinned() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.add(vec![pool.func("erf", vec![x]), pool.func("erfc", vec![x])]);
        let r = range_of(e, &pool, &[(x, -1.0, 1.0)], 8);
        // The polynomial parts cancel *exactly* — that is the Taylor-model
        // property under test. The two remainder intervals do not cancel
        // (negating a model negates a symmetric interval to itself), so the
        // radius is twice `erf`'s own remainder and is not expected to vanish.
        assert!(r.contains(1.0), "erf+erfc: {r:?}");
        assert!(
            (r.mid_f64() - 1.0).abs() < 1e-25,
            "erf+erfc polynomial parts did not cancel: {r:?}"
        );
        assert!(r.rad_f64() < 0.1, "erf remainder is not usable: {r:?}");

        for &p in &[0.0_f64, 0.5, -0.75, 2.0, -3.25] {
            let t = Float::with_val(P, p);
            for name in ["erf", "erfc"] {
                let r = tm_range(name, p, p, 6).unwrap();
                assert_encloses(name, &r, &t, "at a degenerate box");
                assert!(r.rad_f64() < 1e-20, "{name}({p}) radius {}", r.rad_f64());
            }
        }
    }

    /// Randomised sweep for `erf`, 200 boxes over `[-8, 8]`.
    #[test]
    fn erf_random_box_sweep() {
        let mut state: u64 = 0xdead_beef_1234;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut checked = 0usize;
        for _ in 0..200 {
            let c = (next() - 0.5) * 16.0;
            let w = next().powi(3) * 3.0;
            let (lo, hi) = (c - w, c + w);
            let r = match tm_range("erf", lo, hi, 8) {
                Ok(r) => r,
                Err(_) => continue,
            };
            checked += 1;
            for t in samples(lo, hi, 40) {
                assert_encloses("erf", &r, &t, &format!("on [{lo}, {hi}]"));
            }
        }
        assert!(checked > 150, "only {checked}/200 boxes produced a bound");
    }

    /// `erf` must beat the trivial `[-1, 1]` bound on a box where it varies
    /// slowly, or the rule is not worth having.
    #[test]
    fn erf_enclosure_is_tight_where_the_gaussian_is_small() {
        let r = tm_range("erf", 2.0, 2.5, 10).unwrap();
        let span = (ref_val("erf", &Float::with_val(P + 64, 2.5))
            - ref_val("erf", &Float::with_val(P + 64, 2.0)))
        .abs()
        .to_f64();
        assert!(
            width_of(&r) <= 2.0 * span,
            "erf on [2,2.5]: width {} against a true range of {span}",
            width_of(&r)
        );
    }

    /// The rules are Taylor models, not point evaluators: they compose with
    /// the rest of the algebra, over more than one variable.
    #[test]
    fn inverse_hyperbolics_compose() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // asinh(x² + y) on [0,1]×[1,2]
        let arg = pool.add(vec![pool.mul(vec![x, x]), y]);
        let e = pool.func("asinh", vec![arg]);
        let r = range_of(e, &pool, &[(x, 0.0, 1.0), (y, 1.0, 2.0)], 6);
        for i in 0..=20 {
            for j in 0..=20 {
                let a = Float::with_val(P + 64, i as f64 / 20.0);
                let b = Float::with_val(P + 64, 1.0 + j as f64 / 20.0);
                let truth = (a.clone() * a + b).asinh();
                assert!(
                    r.lo() <= truth && truth <= r.hi(),
                    "asinh(x²+y) escaped {r:?} at ({i},{j})"
                );
            }
        }

        // atanh(x/2) is odd, so its enclosure over a symmetric box must be
        // symmetric — a check the per-monomial poly bound cannot fake.
        let half = pool.mul(vec![x, pool.pow(pool.integer(2_i32), pool.integer(-1_i32))]);
        let r = range_of(pool.func("atanh", vec![half]), &pool, &[(x, -1.0, 1.0)], 8);
        assert!(r.mid_f64().abs() < 1e-25, "not symmetric: {r:?}");
        let truth = Float::with_val(P + 64, 0.5).atanh();
        assert!(truth <= r.hi() && -truth.clone() >= r.lo(), "{r:?}");
    }

    #[test]
    fn unsupported_function_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("gamma", vec![x]);
        let boxes = vec![(x, f(1.0), f(2.0))];
        let err = taylor_range(e, &pool, &boxes, 6, P).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-001");
    }

    #[test]
    fn unbound_symbol_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let e = pool.add(vec![x, y]);
        let boxes = vec![(x, f(0.0), f(1.0))];
        let err = taylor_range(e, &pool, &boxes, 6, P).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-002");
    }

    #[test]
    fn two_variable_model_encloses_samples() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // sin(x)·exp(y) on [0,1]×[0,1]
        let e = pool.mul(vec![pool.func("sin", vec![x]), pool.func("exp", vec![y])]);
        let r = range_of(e, &pool, &[(x, 0.0, 1.0), (y, 0.0, 1.0)], 6);
        for i in 0..=40 {
            for j in 0..=40 {
                let a = i as f64 / 40.0;
                let b = j as f64 / 40.0;
                assert!(r.contains(a.sin() * b.exp()), "escaped at ({a},{b})");
            }
        }
    }

    #[test]
    fn degenerate_box_is_point_evaluation() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("exp", vec![x]);
        let r = range_of(e, &pool, &[(x, 1.0, 1.0)], 6);
        // e at higher precision than the enclosure: the f64 constant differs
        // from the true value by far more than this ball's radius.
        let truth = Float::with_val(P + 64, 1.0f64).exp();
        assert!(r.lo() <= truth && truth <= r.hi());
        assert!(r.rad_f64() < 1e-25);
    }

    #[test]
    fn invalid_box_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let boxes = vec![(x, f(2.0), f(1.0))];
        let err = taylor_range(x, &pool, &boxes, 6, P).unwrap_err();
        assert_eq!(crate::errors::AlkahestError::code(&err), "E-VALIDATED-005");
    }
}
