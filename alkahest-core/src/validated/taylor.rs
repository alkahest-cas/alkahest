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
            // dᵏ/dxᵏ sin = sin, cos, -sin, -cos ; cos shifts by one.
            let phase = if is_sin { k % 4 } else { (k + 1) % 4 };
            let base = match phase {
                0 => s.clone(),
                1 => co.clone(),
                2 => -s.clone(),
                _ => -co.clone(),
            };
            let base = if is_sin { base } else { -base };
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
