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
use rug::{Complete, Float, Integer, Rational};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

type Result<T> = std::result::Result<T, ValidatedError>;

/// Exponent vector of a monomial, one entry per box variable.
pub type MultiIndex = Vec<u32>;

/// Highest Taylor order accepted.  Beyond this the coefficient count and the
/// `(p+1)!` remainder scaling stop buying accuracy.
pub const MAX_ORDER: usize = 24;

/// `B_n / n!` as an exact rational, for the Euler–Maclaurin corrections in
/// `hurwitz_zeta_ints`.
///
/// The Bernoulli numbers themselves come from the elementary recurrence
/// `Σ_{j=0}^{m} C(m+1, j)·B_j = 0` (`m ≥ 1`), i.e.
/// `B_m = −(1/(m+1))·Σ_{j<m} C(m+1, j)·B_j`, with `B₀ = 1` — the convention
/// with `B₁ = −½`, which the recurrence produces on its own.  Everything is
/// rational arithmetic, so these are *exact*: nothing about the remainder
/// bound rests on a tabulated decimal.
///
/// Computed once and memoised; `BERNOULLI_MAX` covers twice the Euler–Maclaurin
/// term cap plus the two extra indices the error bound reads.
fn bernoulli_over_factorial(n: usize) -> Rational {
    /// Highest Bernoulli index tabulated.
    const BERNOULLI_MAX: usize = 96;
    static TABLE: OnceLock<Vec<Rational>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut b: Vec<Rational> = Vec::with_capacity(BERNOULLI_MAX + 1);
        b.push(Rational::from(1));
        for m in 1..=BERNOULLI_MAX {
            let mut acc = Rational::new();
            for (j, bj) in b.iter().enumerate().take(m) {
                let c = Integer::from(m as u32 + 1).binomial(j as u32);
                acc += Rational::from(c) * bj;
            }
            acc /= Integer::from(m as u32 + 1);
            b.push(-acc);
        }
        // Divide through by n! once, here, so the caller never handles the
        // (astronomically large) numerator and denominator separately.
        b.into_iter()
            .enumerate()
            .map(|(n, bn)| bn / Integer::factorial(n as u32).complete())
            .collect()
    });
    table
        .get(n)
        .cloned()
        .unwrap_or_else(|| unreachable!("Bernoulli index {n} exceeds the table"))
}

/// `v > 0`, with NaN answering **false**.
///
/// The guards below are all of the form "refuse unless strictly positive", and
/// that has to keep refusing on a NaN endpoint. Writing them as `v <= 0` would
/// invert exactly that case and let a NaN through into a certificate, so the
/// comparison is spelled out through `partial_cmp`.
fn strictly_positive(v: &Float) -> bool {
    matches!(v.partial_cmp(&0), Some(std::cmp::Ordering::Greater))
}

/// Keep the smaller of `slot` and `cand`, ignoring non-finite candidates.
///
/// Used where several rigorous upper bounds on the same remainder are tried
/// (one per Cauchy radius): a minimum of valid upper bounds is a valid upper
/// bound, so the choice among them is a tightness question only.
fn keep_smaller(slot: &mut Option<Float>, cand: Float) {
    if !cand.is_finite() {
        return;
    }
    match slot {
        Some(best) if *best <= cand => {}
        _ => *slot = Some(cand),
    }
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

    /// `C·q^{p+1}/(1−q)` — the tail `Σ_{k>p} |aₖ|·|δ|ᵏ` of a series whose
    /// coefficients admit a **geometric** majorant `|aₖ| ≤ C/ρᵏ`, with
    /// `q = |δ|/ρ`.
    ///
    /// Sibling of [`TaylorModel::series_tail`], which majorises by `1/(kρᵏ)`
    /// instead; the extra `1/k` there is not available for every function, and
    /// where it is not this is the bound that applies.  `t ↦ t^{p+1}/(1−t)` is
    /// increasing on `[0,1)`, so rounding `q` **up** and `1−q` **down** rounds
    /// the result up.  `None` when `q ≥ 1` — outside the disc of convergence
    /// the majorant says nothing and the caller must fall back on Lagrange.
    fn geometric_tail(
        c: &ArbBall,
        delta_mag: &Float,
        rho_lo: &Float,
        p1: usize,
        prec: u32,
    ) -> Option<Float> {
        if !strictly_positive(rho_lo) || !delta_mag.is_finite() {
            return None;
        }
        let q = ub(&Self::div_ball(&from_float(delta_mag, prec), &from_float(rho_lo, prec)).ok()?);
        let one_minus_q = lb(&(ArbBall::from_f64(1.0, prec) - from_float(&q, prec)));
        if !strictly_positive(&one_minus_q) {
            return None;
        }
        let num = from_float(&q, prec).powi(p1 as i64) * symmetric(&mag(c), prec);
        let out = ub(&Self::div_ball(&num, &from_float(&one_minus_q, prec)).ok()?);
        out.is_finite().then_some(out)
    }

    // ── Bessel Jν ────────────────────────────────────────────────────────

    /// `Jν(self)` for integer order `ν ≥ 0`.  Entire, so there is no domain
    /// guard.
    ///
    /// **Both the coefficients and the remainder come from one identity.**
    /// The three-term derivative relation
    /// `2·Jν′(x) = J_{ν−1}(x) − J_{ν+1}(x)` (and `J₀′ = −J₁`, its `ν = 0`
    /// case, since `J₋₁ = −J₁`) iterates by Pascal's rule into
    ///
    /// ```text
    /// Jν⁽ⁿ⁾(x) = 2⁻ⁿ · Σ_{j=0}^{n} (−1)ʲ · C(n,j) · J_{ν−n+2j}(x).
    /// ```
    ///
    /// *Induction.*  True at `n = 0`.  Differentiating the `n`-th line term by
    /// term and applying `2Jμ′ = J_{μ−1} − J_{μ+1}` to each `J_{ν−n+2j}`
    /// produces `2^{−(n+1)} Σ_j (−1)ʲ C(n,j) [J_{ν−n−1+2j} − J_{ν−n+1+2j}]`;
    /// re-indexing the second sum by `j → j−1` merges the two into
    /// `Σ_j (−1)ʲ [C(n,j) + C(n,j−1)] J_{ν−(n+1)+2j}`, and
    /// `C(n,j) + C(n,j−1) = C(n+1,j)`. ∎
    ///
    /// **Remainder.**  For every integer `m` and every real `x`,
    ///
    /// ```text
    /// J_m(x) = (1/π)·∫₀^π cos(mθ − x·sin θ) dθ    ⟹    |J_m(x)| ≤ 1,
    /// ```
    ///
    /// the integrand being a cosine and the interval having length `π`.  Put
    /// that into the identity above: the binomial coefficients sum to `2ⁿ`,
    /// which the `2⁻ⁿ` exactly cancels, so
    ///
    /// ```text
    /// |Jν⁽ⁿ⁾(x)| ≤ 1        for every real x and every order n,
    /// ```
    ///
    /// and hence `|Jν⁽ᵖ⁺¹⁾(ξ)|/(p+1)! ≤ 1/(p+1)!` uniformly — the same
    /// remainder `sin` and `cos` get, for the same reason (a uniform bound on
    /// *every* derivative), and with no appeal to monotonicity anywhere.  That
    /// last point is what makes this sound where the endpoint hull in
    /// [`crate::ball::ArbBall::bessel_jn`] was not: `Jν` oscillates, and
    /// nothing here supposes otherwise.
    ///
    /// A Cauchy estimate against the entire-function growth
    /// (`|Jν(z)| ≤ |z/2|^ν e^{|Im z|}/ν!` from the Poisson integral) is also
    /// available but is never sharper: minimised over the circle radius it
    /// gives `≈ √(2πn)/n!`, a factor `√(2πn)` *worse* than the bound above.
    /// So this rule uses the elementary one alone.
    pub fn bessel_j(&self, nu: i32) -> Result<Self> {
        self.check_finite("bessel argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "bessel argument".into(),
            });
        }
        let p = self.order;
        // J_{ν−p} … J_{ν+p}, each correctly rounded by MPFR and then
        // re-rounded outward into a ball.
        let work = self.prec + 32;
        let jvals: Vec<ArbBall> = (0..=2 * p)
            .map(|i| {
                let order = nu - (p as i32) + (i as i32);
                let mut v = Float::with_val(work, &m0);
                v.jn_mut(order);
                from_float(&v, self.prec)
            })
            .collect();

        let mut a = Vec::with_capacity(p + 1);
        for k in 0..=p {
            let mut acc = ArbBall::from_f64(0.0, self.prec);
            for j in 0..=k {
                let binom = Integer::from(k).binomial(j as u32);
                let term = ArbBall::from_integer(&binom, self.prec) * jvals[p - k + 2 * j].clone();
                acc = if j % 2 == 0 { acc + term } else { acc - term };
            }
            // 2ᵏ·k! — exact, so the division only costs the ball's own
            // rounding.
            let scale = ArbBall::from_integer(
                &((Integer::from(1) << (k as u32)) * Integer::factorial(k as u32).complete()),
                self.prec,
            );
            a.push(Self::div_ball(&acc, &scale)?);
        }

        let fact = Self::factorial(self.order + 1, self.prec);
        let scale = Self::div_ball(&ArbBall::from_f64(1.0, self.prec), &fact)?;
        let radius = ub(&(scale
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("bessel result")?;
        Ok(out)
    }

    // ── digamma / gamma ──────────────────────────────────────────────────

    /// `ζ(s, a) = Σ_{k≥0} (a+k)^{-s}` for every integer `s` in `2..=s_max`,
    /// as rigorous balls, for real `a > 0`.  Index `i` holds `s = i + 2`.
    ///
    /// The Hurwitz zeta is what the Taylor coefficients of `ψ` (and, through
    /// `ψ`, of `Γ`) *are*: `ψ⁽ᵏ⁾(a) = (−1)^{k+1}·k!·ζ(k+1, a)`.  Direct
    /// summation is hopeless — the tail of `ζ(2, a)` decays like `1/N` — so
    /// this is Euler–Maclaurin applied to the tail after `SHIFT` exact terms.
    ///
    /// With `A = a + SHIFT` and `f(t) = (a+t)^{-s}`, whose derivatives are
    /// `f⁽ʲ⁾(t) = (−1)ʲ·(s)_j·(a+t)^{−s−j}`:
    ///
    /// ```text
    /// ζ(s,a) = Σ_{k<SHIFT} (a+k)^{-s}
    ///        + A^{1-s}/(s-1)                       [ ∫_SHIFT^∞ f ]
    ///        + A^{-s}/2                            [ ½·f(SHIFT)  ]
    ///        + Σ_{k=1}^{K} (B_{2k}/(2k)!)·(s)_{2k-1}·A^{1-s-2k}
    ///        + R_K.
    /// ```
    ///
    /// **The remainder.**  `R_K` is an integral of the periodic Bernoulli
    /// function against `f⁽²ᴷ⁺²⁾`.  Two facts bound it without any appeal to
    /// tabulated constants: `max_{[0,1]} |B_{2m}(u)| = |B_{2m}|` (immediate
    /// from the Fourier series `B̃_{2m}(u) = (−1)^{m+1}·2·(2m)!·(2π)^{−2m}·
    /// Σ_{j≥1} cos(2πju)/j^{2m}`, whose terms all peak together at `u = 0`),
    /// and `f⁽²ᴷ⁺²⁾` has one sign on `[SHIFT, ∞)` so `∫|f⁽²ᴷ⁺²⁾| =
    /// |f⁽²ᴷ⁺¹⁾(SHIFT)|`.  Hence
    ///
    /// ```text
    /// |R_K| ≤ 2·(|B_{2K+2}|/(2K+2)!)·(s)_{2K+1}·A^{−s−2K−1},
    /// ```
    ///
    /// i.e. **twice the first omitted term**.  The factor 2 is deliberate
    /// slack: the two standard forms of the Euler–Maclaurin remainder differ
    /// by whether `B̃_{2m}` or `B̃_{2m} − B_{2m}` appears, and `2|B_{2m}|`
    /// dominates both, so the bound holds under either convention rather than
    /// depending on which one is quoted.  The series is asymptotic, so the
    /// loop stops at the first term that is small enough *or* that stops
    /// shrinking, and reports twice that term as the radius.
    fn hurwitz_zeta_ints(a: &ArbBall, s_max: usize, prec: u32) -> Result<Vec<ArbBall>> {
        /// Exact terms taken before the asymptotic tail.  100 puts `A ≥ 100`,
        /// where the correction terms fall off by roughly `(2π·100)⁻²` each.
        const SHIFT: usize = 100;
        /// Cap on Euler–Maclaurin terms; the series diverges eventually, and
        /// the loop normally stops long before this.
        const MAX_TERMS: usize = 40;

        let a_lo = lb(a);
        if !strictly_positive(&a_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "Hurwitz zeta needs a strictly positive second argument".into(),
            });
        }
        let one = ArbBall::from_f64(1.0, prec);
        let big = a.clone() + ArbBall::from_f64(SHIFT as f64, prec);
        let inv_big = Self::div_ball(&one, &big)?;
        let inv_big2 = inv_big.clone() * inv_big.clone();
        // Accumulate every head sum in one pass over `k`: `(a+k)^{-s}` for
        // consecutive `s` is one multiplication apart, so this costs one
        // division and `s_max` multiplications per term rather than a fresh
        // binary exponentiation for each `(k, s)` pair.
        let count = s_max.saturating_sub(1);
        let mut heads = vec![ArbBall::from_f64(0.0, prec); count];
        for k in 0..SHIFT {
            let t = a.clone() + ArbBall::from_f64(k as f64, prec);
            let inv = Self::div_ball(&one, &t)?;
            let mut pw = inv.clone() * inv.clone();
            for head in heads.iter_mut() {
                *head = head.clone() + pw.clone();
                pw = pw * inv.clone();
            }
        }

        let mut out = Vec::with_capacity(count);
        let mut a_neg_s = inv_big.clone() * inv_big.clone();
        for s in 2..=s_max {
            let head = heads[s - 2].clone();
            if s > 2 {
                a_neg_s = a_neg_s.clone() * inv_big.clone();
            }
            let a_neg_s = a_neg_s.clone();
            let integral = Self::div_ball(
                &(a_neg_s.clone() * big.clone()),
                &ArbBall::from_f64((s - 1) as f64, prec),
            )?;
            let half = Self::div_ball(&a_neg_s, &ArbBall::from_f64(2.0, prec))?;
            let mut acc = head + integral + half;

            // termₖ = (B_{2k}/(2k)!)·(s)_{2k-1}·A^{1-s-2k}.
            //         `poch` is (s)_{2k-1}, `pow` is A^{1-s-2k}.
            // k = 1: (s)_1 = s and A^{1-s-2} = A^{-(s+1)}; each step multiplies
            // the Pochhammer by two more factors and the power by A^{-2}.
            let mut poch = Integer::from(s as u32);
            let mut pow = inv_big.powi((s + 1) as i64);
            // A term below this fraction of the partial sum is past the
            // working precision, so there is nothing left to gain by adding it.
            let mut small = mag(&acc);
            small >>= prec.saturating_sub(4);
            let mut prev: Option<Float> = None;
            let mut radius = Float::new(prec);
            for k in 1..=MAX_TERMS {
                if k > 1 {
                    // (s)_{2k-1} = (s)_{2k-3}·(s+2k-3)·(s+2k-2)
                    poch *= Integer::from(s + 2 * k - 3);
                    poch *= Integer::from(s + 2 * k - 2);
                    pow = pow.clone() * inv_big2.clone();
                }
                let term = ArbBall::from_rational(&bernoulli_over_factorial(2 * k), prec)
                    * ArbBall::from_integer(&poch, prec)
                    * pow.clone();
                let tm = mag(&term);
                let stop_small = tm <= small;
                let stop_diverging = prev.as_ref().is_some_and(|p| &tm >= p);
                if stop_small || stop_diverging || k == MAX_TERMS {
                    radius = ub(&(symmetric(&tm, prec) * ArbBall::from_f64(2.0, prec)));
                    break;
                }
                acc = acc + term;
                prev = Some(tm);
            }
            acc.rad += radius;
            if !is_finite(&acc) {
                return Err(ValidatedError::NotFinite {
                    what: format!("ζ({s}, a)"),
                });
            }
            out.push(acc);
        }
        Ok(out)
    }

    /// `digamma(self)`.  Refuses unless the argument enclosure lies strictly
    /// inside `(0, ∞)`.
    ///
    /// **Domain.**  `ψ` has a simple pole at every non-positive integer, and
    /// between two of them the coefficient machinery below (a Hurwitz zeta
    /// summed over `a, a+1, a+2, …`) does not converge at all, so the guard is
    /// `lb(arg) > 0` and nothing weaker.  A box like `[−2.5, −2.1]` sits
    /// between poles and *is* a domain of analyticity, but it is refused
    /// rather than answered by a reflection formula nobody has written here.
    ///
    /// **Coefficients.**  `ψ⁽ᵏ⁾(x) = (−1)^{k+1}·k!·ζ(k+1, x)` — differentiate
    /// `ψ(x) = −γ + Σ_{n≥0} [1/(n+1) − 1/(x+n)]` termwise, which is legitimate
    /// because the differentiated series converges locally uniformly on
    /// `x > 0`.  So `aₖ = ψ⁽ᵏ⁾(m₀)/k! = (−1)^{k+1}·ζ(k+1, m₀)` exactly, with
    /// no recurrence and no cancellation, and `a₀ = ψ(m₀)` from MPFR.
    ///
    /// **Remainder.**  Two bounds, whichever is smaller:
    ///
    /// * *Lagrange.*  `|ψ⁽ᵖ⁺¹⁾(ξ)|/(p+1)! = ζ(p+2, ξ)`, and `ζ(s, ξ)` is
    ///   manifestly decreasing in `ξ` (every term `(ξ+k)^{-s}` is), so its
    ///   supremum over the enclosure sits at the lower endpoint `L`.  There
    ///   `ζ(s, L) ≤ L^{-s} + ∫₀^∞ (L+t)^{-s} dt = L^{-s} + L^{1-s}/(s−1)`,
    ///   comparing `(L+k)^{-s} ≤ ∫_{k-1}^{k}(L+t)^{-s} dt` term by term.  With
    ///   `s = p+2` that is `L^{-(p+2)} + L^{-(p+1)}/(p+1)`.
    /// * *Geometric tail.*  The same estimate gives
    ///   `|aₖ| = ζ(k+1, m₀) ≤ m₀^{-(k+1)} + m₀^{-k}/k ≤ C·m₀^{-k}` with
    ///   `C = 1/m₀ + 1/(p+1)` for every `k ≥ p+1`, so
    ///   `geometric_tail` applies with `ρ = m₀` — the distance
    ///   from `m₀` to the pole at the origin, which is exactly the radius of
    ///   convergence.
    ///
    /// Neither uses monotonicity of `ψ` itself; the second bound's `ρ` is a
    /// statement about where the poles are, not about the shape of the graph.
    pub fn digamma(&self) -> Result<Self> {
        self.check_finite("digamma argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let arg_lo = lb(&arg);
        if !strictly_positive(&arg_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "digamma of an argument whose enclosure reaches 0 or below (poles sit at every non-positive integer)".into(),
            });
        }
        let c = from_float(&m0, self.prec);
        let p1 = self.order + 1;
        let mut a = Vec::with_capacity(p1);
        let mut psi = Float::with_val(self.prec + 32, &m0);
        psi.digamma_mut();
        a.push(from_float(&psi, self.prec));
        if self.order >= 1 {
            let zetas = Self::hurwitz_zeta_ints(&c, self.order + 1, self.prec)?;
            for k in 1..=self.order {
                let z = zetas[k - 1].clone();
                a.push(if k % 2 == 1 { z } else { -z });
            }
        }

        // Lagrange: ζ(p+2, L) ≤ L^{-(p+2)} + L^{-(p+1)}/(p+1).
        let lo_ball = from_float(&arg_lo, self.prec);
        let one = ArbBall::from_f64(1.0, self.prec);
        let sup = Self::div_ball(&one, &lo_ball.powi((p1 + 1) as i64))?
            + Self::div_ball(
                &one,
                &(lo_ball.powi(p1 as i64) * ArbBall::from_f64(p1 as f64, self.prec)),
            )?;
        let lagrange = ub(&(sup
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(self.prec),
                prec: self.prec,
            }));
        // Geometric tail with ρ = m₀ and C = 1/m₀ + 1/(p+1).
        let tail = (|| {
            let cc = Self::div_ball(&one, &c).ok()?
                + Self::div_ball(&one, &ArbBall::from_f64(p1 as f64, self.prec)).ok()?;
            Self::geometric_tail(&cc, &mag(&d), &lb(&c), p1, self.prec)
        })();
        let radius = Self::tighter(lagrange, tail);
        let out = delta.compose(&a, &radius);
        out.check_finite("digamma result")?;
        Ok(out)
    }

    /// Upper bound on `max_{|z−x| = r} |Γ(z)|` for real `x` ranging over
    /// `[lo, hi]`, valid whenever `lo − r > 0`.
    ///
    /// Two elementary facts, both from the Euler integral:
    ///
    /// * `|Γ(u+iv)| = |∫₀^∞ t^{u−1+iv} e^{−t} dt| ≤ ∫₀^∞ t^{u−1} e^{−t} dt
    ///   = Γ(u)` for `u > 0`, because `|t^{iv}| = 1` on `t > 0`;
    /// * `Γ″(u) = ∫₀^∞ t^{u−1}(ln t)² e^{−t} dt > 0`, so `Γ` is **convex** on
    ///   `(0, ∞)` and its maximum over an interval is at an endpoint.
    ///
    /// A point of a circle `|z − x| = r` with `x ∈ [lo, hi]` has
    /// `Re z ∈ [lo − r, hi + r]`, so `|Γ(z)| ≤ max(Γ(lo−r), Γ(hi+r))`.  No
    /// monotonicity of `Γ` is assumed — it has a minimum at `x ≈ 1.4616`, and
    /// convexity is what covers a box straddling it.
    fn gamma_circle_bound(lo: &Float, hi: &Float, r: &Float, prec: u32) -> Option<ArbBall> {
        let work = prec + 32;
        // `lo − r` **rounded down** and `hi + r` **rounded up**: the strip
        // whose maximum is being taken has to contain the true one, and a
        // round-to-nearest here would shave an ulp off the end where Γ is
        // steepest, which is the one direction a certificate must not move in.
        let rb = from_float(r, work);
        let left = lb(&(from_float(lo, work) - rb.clone()));
        if !matches!(left.partial_cmp(&0), Some(std::cmp::Ordering::Greater)) {
            return None;
        }
        let right = ub(&(from_float(hi, work) + rb));
        let ga = from_float(&Float::with_val(work, left).gamma(), prec);
        let gb = from_float(&Float::with_val(work, right).gamma(), prec);
        let out = if ub(&ga) > ub(&gb) { ga } else { gb };
        is_finite(&out).then_some(out)
    }

    /// `gamma(self)`.  Refuses unless the argument enclosure lies strictly
    /// inside `(0, ∞)`.
    ///
    /// **Domain.**  `Γ` has a pole at every non-positive integer.  On the
    /// strips between them it is analytic, but both the coefficients (via
    /// `ψ`, hence via a Hurwitz zeta needing `a > 0`) and the remainder (via
    /// `|Γ(u+iv)| ≤ Γ(u)`, needing `u > 0`) are written for the positive axis
    /// only, so anything reaching `0` is refused.
    ///
    /// **Coefficients.**  From `Γ′ = ψ·Γ`, Leibniz gives
    /// `Γ⁽ⁿ⁺¹⁾ = Σ_{j=0}^{n} C(n,j)·ψ⁽ʲ⁾·Γ⁽ⁿ⁻ʲ⁾`; dividing by `(n+1)!` turns
    /// the binomials into a plain convolution of Taylor coefficients,
    ///
    /// ```text
    /// c_{n+1} = (1/(n+1))·Σ_{j=0}^{n} d_j · c_{n−j},
    /// ```
    ///
    /// with `cₖ = Γ⁽ᵏ⁾(m₀)/k!`, `c₀ = Γ(m₀)`, and `dⱼ = ψ⁽ʲ⁾(m₀)/j!` — which
    /// is `d₀ = ψ(m₀)` and `dⱼ = (−1)^{j+1} ζ(j+1, m₀)`, the very numbers
    /// [`TaylorModel::digamma`] expands with.  This is an identity between
    /// derivatives at the single point `m₀`, so ball arithmetic runs it
    /// without any interval widening.
    ///
    /// **Remainder.**  `Γ` is analytic on `Re z > 0`, so Cauchy's estimate
    /// holds for every radius `r` with `L − r > 0`:
    ///
    /// ```text
    /// |Γ⁽ᵖ⁺¹⁾(ξ)|/(p+1)!  ≤  max_{|z−ξ|=r} |Γ(z)| / r^{p+1}
    ///                     ≤  max(Γ(L−r), Γ(U+r)) / r^{p+1},
    /// ```
    ///
    /// by `gamma_circle_bound`.  **Every `r` gives a valid
    /// bound**, so the candidates tried below are a tightness choice only and
    /// cannot make the result unsound.  The same estimate at the expansion
    /// point, `|cₖ| ≤ max(Γ(m₀−r), Γ(m₀+r))/rᵏ`, feeds
    /// `geometric_tail`; that form is usually several orders
    /// tighter because it never takes a supremum over the whole enclosure.
    /// The smallest of all of them is kept, and a minimum of valid upper
    /// bounds is a valid upper bound.
    pub fn gamma(&self) -> Result<Self> {
        self.check_finite("gamma argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        let arg_lo = lb(&arg);
        let arg_hi = ub(&arg);
        if !strictly_positive(&arg_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "gamma of an argument whose enclosure reaches 0 or below (poles sit at every non-positive integer)".into(),
            });
        }
        let prec = self.prec;
        let c = from_float(&m0, prec);
        let p = self.order;
        let p1 = p + 1;

        // dⱼ = ψ⁽ʲ⁾(m₀)/j!, j = 0..p−1 (the convolution only ever reads that far).
        let mut dvec = Vec::with_capacity(p.max(1));
        let mut psi = Float::with_val(prec + 32, &m0);
        psi.digamma_mut();
        dvec.push(from_float(&psi, prec));
        if p >= 1 {
            let zetas = Self::hurwitz_zeta_ints(&c, p.max(2), prec)?;
            for j in 1..p {
                let z = zetas[j - 1].clone();
                dvec.push(if j % 2 == 1 { z } else { -z });
            }
        }
        let mut a = Vec::with_capacity(p + 1);
        a.push(from_float(&Float::with_val(prec + 32, &m0).gamma(), prec));
        for n in 0..p {
            let mut acc = ArbBall::from_f64(0.0, prec);
            for j in 0..=n {
                acc = acc + dvec[j].clone() * a[n - j].clone();
            }
            a.push(Self::div_ball(
                &acc,
                &ArbBall::from_f64((n + 1) as f64, prec),
            )?);
        }

        let dmag = mag(&d);
        let delta_pow = ArbBall {
            mid: delta.delta_pow(&d),
            rad: Float::new(prec),
            prec,
        };
        let mut best: Option<Float> = None;
        let mut keep = |cand: Option<Float>| {
            if let Some(v) = cand {
                if v.is_finite() && (best.is_none() || best.as_ref().is_some_and(|b| &v < b)) {
                    best = Some(v);
                }
            }
        };
        // Radii as fractions of the distance to the pole at the origin. The
        // Lagrange form needs r < L; the tail form needs |δ| < r < m₀.
        for frac in [0.99_f64, 0.9, 0.75, 0.5, 0.25, 0.1] {
            let r_lag = Float::with_val(prec, &arg_lo * frac);
            if let Some(m) = Self::gamma_circle_bound(&arg_lo, &arg_hi, &r_lag, prec) {
                let denom = from_float(&r_lag, prec).powi(p1 as i64);
                if let Ok(scale) = Self::div_ball(&m, &denom) {
                    keep(Some(ub(&(scale * delta_pow.clone()))));
                }
            }
            let r_tail = Float::with_val(prec, &m0 * frac);
            if let Some(m) = Self::gamma_circle_bound(&m0, &m0, &r_tail, prec) {
                keep(Self::geometric_tail(&m, &dmag, &r_tail, p1, prec));
            }
        }
        let radius = best.ok_or_else(|| ValidatedError::NotFinite {
            what: "gamma remainder bound".into(),
        })?;
        let out = delta.compose(&a, &radius);
        out.check_finite("gamma result")?;
        Ok(out)
    }

    // ── Lambert W ────────────────────────────────────────────────────────

    /// `lambert_w(self)` — the principal branch `W₀`.  Refuses unless the
    /// argument enclosure lies strictly above `−1/e`.
    ///
    /// **Domain.**  `W₀` is real on `[−1/e, ∞)` and has a square-root branch
    /// point at `−1/e`, where every derivative is unbounded, so no Taylor
    /// remainder exists there.  The guard is not a comparison against a
    /// rounded `−1/e`: it is `W₀(L) > −1`, checked on the *certified* bracket
    /// from [`crate::ball::ArbBall::lambert_w0`], which refuses on its own for
    /// any argument left of the branch point.
    ///
    /// **Coefficients.**  Writing `w = W₀(x)`, `x = w·eʷ` gives
    /// `dx/dw = (1+w)eʷ` and hence `W₀′ = e^{−w}/(1+w)`.  Induction on that
    /// yields the classical closed form
    ///
    /// ```text
    /// W₀⁽ⁿ⁾(x) = e^{−n·w} · pₙ(w) / (1+w)^{2n−1},
    /// p₁ = 1,   p_{n+1}(w) = (1+w)·pₙ′(w) − (n·w + 3n − 1)·pₙ(w),
    /// ```
    ///
    /// with `pₙ` a polynomial of degree `n−1` and **integer** coefficients (so
    /// they are computed exactly here, not in floating point).  *Proof of the
    /// step*: differentiate the `n`-th expression with respect to `x` by the
    /// chain rule, multiplying by `dw/dx = e^{−w}/(1+w)`; collecting the three
    /// resulting terms over the common denominator `(1+w)^{2n+1}` gives
    /// exactly the stated recurrence. ∎
    ///
    /// **Remainder.**  With `wL = W₀(L)` and `wU = W₀(U)` — and `W₀` is
    /// increasing, because `W₀′ = e^{−w}/(1+w) > 0` for `w > −1`, so `w`
    /// ranges over `[wL, wU]` as `ξ` ranges over `[L, U]` — every factor of
    /// the closed form is bounded separately:
    ///
    /// * `e^{−(p+1)w} ≤ e^{−(p+1)·wL}`, since it decreases in `w`;
    /// * `(1+w)^{2p+1} ≥ (1+wL)^{2p+1}`, since `1+w > 0` and it increases;
    /// * `|p_{p+1}(w)| ≤ Σ_j |coefficient_j| · max(|wL|, |wU|)^j`, the
    ///   triangle inequality — no cancellation is claimed — or the ball
    ///   -arithmetic Horner value of `p_{p+1}` over the same range, whichever
    ///   is smaller; both enclose it, so the minimum does too.
    ///
    /// The three are monotone in *opposite* directions, so evaluating them all
    /// at `wL` is very loose over a wide range.  The `w` range is therefore
    /// split into panels and the largest panel bound kept, each panel using
    /// its own left end — which is what the two monotonicity facts above
    /// license.  A finer split can only shrink the answer, so the panel count
    /// is a tightness knob and not a soundness one; the panel boundaries are
    /// forced to `wL` and `wU` at the ends so no sliver of the range escapes
    /// the supremum.
    ///
    /// This is the loosest of the rules near its boundary, and honestly so:
    /// `W₀⁽ⁿ⁾` really does blow up like `(x + 1/e)^{1/2−n}` at the branch
    /// point, so a single un-subdivided model on `[−0.36, −0.3]` is useless
    /// while `bound_on_box` converges on the true range in 39 subdivisions.
    ///
    /// The three combine to a bound on `|W₀⁽ᵖ⁺¹⁾(ξ)|/(p+1)!` valid for every
    /// `ξ` in the enclosure at once, which is what a Lagrange remainder needs.
    /// Each bound is a monotonicity statement about an *explicit elementary
    /// factor*, proved on the spot; none is an assumption about `W₀`.
    pub fn lambert_w(&self) -> Result<Self> {
        self.check_finite("lambert_w argument")?;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, self.prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "lambert_w argument".into(),
            });
        }
        let prec = self.prec;
        let domain_err = || {
            ValidatedError::DomainViolation {
            what: "lambert_w of an argument whose enclosure reaches -1/e or below (the principal branch has a branch point there, where every derivative is unbounded)".into(),
        }
        };
        let w_range = arg.lambert_w0().ok_or_else(domain_err)?;
        let w_lo = lb(&w_range);
        let w_hi = ub(&w_range);
        // `1 + W₀(L) > 0` strictly: at the branch point it is 0 and the
        // closed form below divides by its (2n−1)-st power.
        let one_plus_lo = lb(&(from_float(&w_lo, prec) + ArbBall::from_f64(1.0, prec)));
        if !strictly_positive(&one_plus_lo) {
            return Err(domain_err());
        }
        let w0 = from_float(&m0, prec).lambert_w0().ok_or_else(domain_err)?;

        // pₙ, exact integer coefficients, low degree first.
        let p = self.order;
        let mut polys: Vec<Vec<Integer>> = Vec::with_capacity(p + 2);
        polys.push(vec![Integer::from(1)]); // p₁
        for n in 1..=p {
            let prev = &polys[n - 1];
            // (1+w)·p′ − (n·w + 3n − 1)·p
            let deg = prev.len();
            let mut next = vec![Integer::new(); deg + 1];
            for (j, cj) in prev.iter().enumerate() {
                if j >= 1 {
                    // (1+w)·(j·c_j·w^{j-1}) = j·c_j·w^{j-1} + j·c_j·w^j
                    let t = Integer::from(j as u32) * cj.clone();
                    next[j - 1] += t.clone();
                    next[j] += t;
                }
                // −(n·w + 3n − 1)·c_j·w^j
                next[j] -= Integer::from(3 * n as u32 - 1) * cj.clone();
                next[j + 1] -= Integer::from(n as u32) * cj.clone();
            }
            while next.len() > 1 && next.last().is_some_and(|c| c.is_zero()) {
                next.pop();
            }
            polys.push(next);
        }

        let one = ArbBall::from_f64(1.0, prec);
        let mut a = Vec::with_capacity(p + 1);
        a.push(w0.clone());
        let e_neg_w = (-w0.clone()).exp();
        let one_plus_w = one.clone() + w0.clone();
        for n in 1..=p {
            let mut pv = ArbBall::from_f64(0.0, prec);
            for (j, cj) in polys[n - 1].iter().enumerate() {
                pv = pv + ArbBall::from_integer(cj, prec) * w0.powi(j as i64);
            }
            let num = e_neg_w.powi(n as i64) * pv;
            let den = one_plus_w.powi((2 * n - 1) as i64) * Self::factorial(n, prec);
            a.push(Self::div_ball(&num, &den)?);
        }

        // Lagrange bound at order p+1.
        //
        // The three factors of the closed form are each monotone in `w`, but
        // in *different directions*, so bounding all of them at `wL` at once
        // is very loose over a wide `w` range.  Splitting `[wL, wU]` into
        // panels and taking the largest panel bound removes that: on a panel
        // `[wa, wb]` the same three monotonicity facts give
        // `e^{-(p+1)w} ≤ e^{-(p+1)wa}`, `(1+w)^{2p+1} ≥ (1+wa)^{2p+1}` and
        // `|p_{p+1}(w)| ≤ min(Σ|c_j|·max(|wa|,|wb|)^j, |p_{p+1}([wa,wb])|)`,
        // the second `|·|` being ball-arithmetic Horner over the panel, which
        // keeps the coefficients' sign structure where the triangle inequality
        // throws it away.  Both are enclosures of the same quantity, so the
        // smaller is still one.  A finer split can only shrink the answer, so
        // the panel count is a tightness knob and not a soundness one.
        const PANELS: usize = 48;
        let p1 = p + 1;
        let fact_p1 = Self::factorial(p1, prec);
        let width = ub(&(from_float(&w_hi, prec) - from_float(&w_lo, prec)));
        // Panel boundaries.  The two *ends* are forced to `w_lo` and `w_hi`
        // exactly and consecutive panels share a boundary, so the union is
        // `[w_lo, w_hi]` whatever the interior boundaries round to — a sliver
        // left uncovered at either end would be a hole in the supremum.
        let boundary = |i: usize| -> Float {
            if i == 0 {
                w_lo.clone()
            } else if i >= PANELS {
                w_hi.clone()
            } else {
                Float::with_val(
                    prec,
                    &w_lo + Float::with_val(prec, &width * (i as f64 / PANELS as f64)),
                )
            }
        };
        let mut sup = ArbBall::from_f64(0.0, prec);
        for i in 0..PANELS {
            let (u, v) = (boundary(i), boundary(i + 1));
            // `wa` is the panel's *left* end, which is what the two monotone
            // factors below are evaluated at; take the minimum so a
            // non-monotone rounding of the interior boundaries cannot put it
            // above a point of the panel.
            let (wa, wb) = if u <= v { (u, v) } else { (v, u) };
            let span = from_bounds(&wa, &wb, prec);
            let amax = mag(&span);
            let amax_b = from_float(&amax, prec);
            let mut psum = ArbBall::from_f64(0.0, prec);
            for (j, cj) in polys[p].iter().enumerate() {
                let abs = Integer::from(cj.abs_ref());
                psum = psum + ArbBall::from_integer(&abs, prec) * amax_b.powi(j as i64);
            }
            let mut horner = ArbBall::from_f64(0.0, prec);
            for cj in polys[p].iter().rev() {
                horner = horner * span.clone() + ArbBall::from_integer(cj, prec);
            }
            let pbound = {
                let h = mag(&horner);
                let t = mag(&psum);
                symmetric(if h < t { &h } else { &t }, prec)
            };
            // `1 + wa > 0` follows from `1 + wL > 0`, checked above.
            let base = from_float(&wa, prec);
            let num = (-base.clone() * ArbBall::from_f64(p1 as f64, prec)).exp() * pbound;
            let den =
                (ArbBall::from_f64(1.0, prec) + base).powi((2 * p1 - 1) as i64) * fact_p1.clone();
            let cand = Self::div_ball(&num, &den)?;
            if mag(&cand) > mag(&sup) {
                sup = symmetric(&mag(&cand), prec);
            }
        }
        let radius = ub(&(sup
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(prec),
                prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("lambert_w result")?;
        Ok(out)
    }

    // ── Fresnel integrals (3.10.0) ───────────────────────────────────────

    /// `π` as an outward-rounded ball at `prec`.
    fn pi_ball(prec: u32) -> ArbBall {
        from_float(&Float::with_val(prec + 32, rug::float::Constant::Pi), prec)
    }

    /// `(sⁿ, cⁿ)` for `n = 0..count`, where `s(x) = sin(πx²/2)` and
    /// `c(x) = cos(πx²/2)` are the two Fresnel integrands, evaluated at `m₀`.
    ///
    /// `s′ = πx·c` and `c′ = −πx·s`; `x` is linear, so the Leibniz rule has
    /// exactly two terms and iterating gives
    ///
    /// ```text
    /// s⁽ⁿ⁺¹⁾ = π·(m₀·c⁽ⁿ⁾ + n·c⁽ⁿ⁻¹⁾),      c⁽ⁿ⁺¹⁾ = −π·(m₀·s⁽ⁿ⁾ + n·s⁽ⁿ⁻¹⁾).
    /// ```
    ///
    /// This is an identity between derivatives at the single point `m₀`, so
    /// running it in ball arithmetic cannot widen by dependency: there is no
    /// interval to widen.
    fn fresnel_phase_derivs(m0: &ArbBall, count: usize, prec: u32) -> (Vec<ArbBall>, Vec<ArbBall>) {
        let pi = Self::pi_ball(prec);
        let half = ArbBall::from_f64(0.5, prec);
        let arg = m0.clone() * m0.clone() * pi.clone() * half;
        let mut s = vec![arg.sin()];
        let mut c = vec![arg.cos()];
        for n in 0..count.saturating_sub(1) {
            let mut sn = m0.clone() * c[n].clone();
            let mut cn = m0.clone() * s[n].clone();
            if n >= 1 {
                let nb = ArbBall::from_f64(n as f64, prec);
                sn = sn + nb.clone() * c[n - 1].clone();
                cn = cn + nb * s[n - 1].clone();
            }
            s.push(pi.clone() * sn);
            c.push(-(pi.clone() * cn));
        }
        (s, c)
    }

    /// `sup |sin(πz²/2)|` and `sup |cos(πz²/2)|` over `|z − ξ| ≤ r` for every
    /// real `ξ` with `|ξ| ≤ m`, as an outward-rounded ball.
    ///
    /// `|sin w| ≤ cosh(Im w)` and `|cos w| ≤ cosh(Im w)` (from
    /// `|sin(u+iv)|² = sin²u + sinh²v`), and with `w = πz²/2`,
    /// `Im w = π·Re z·Im z`, so `|Im w| ≤ π·(m + r)·r`.
    fn fresnel_phase_bound(m: &Float, r: &Float, prec: u32) -> ArbBall {
        let pi = Self::pi_ball(prec);
        let rb = from_float(r, prec);
        let arg = pi * (from_float(m, prec) + rb.clone()) * rb;
        arg.cosh()
    }

    /// `S(self)` (`sine = true`) or `C(self)`, in the normalised π/2
    /// convention of [`crate::primitive::fresnel`].  Both are entire, so there
    /// is no domain guard.
    ///
    /// **`a₀`** is a rigorous point enclosure from
    /// [`crate::primitive::fresnel::fresnel_pair_ball`] (power series below
    /// `|x| = 6`, DLMF's asymptotic expansion above it, with both truncations
    /// charged to the radius).  **`aₖ` for `k ≥ 1`** is `s⁽ᵏ⁻¹⁾(m₀)/k!` — the
    /// integrand's own derivatives, from [`Self::fresnel_phase_derivs`], since
    /// `S⁽ᵏ⁾ = s⁽ᵏ⁻¹⁾`.
    ///
    /// **Remainder.**  `S` is entire, so Cauchy's estimate holds at *every*
    /// radius `r > 0`:
    ///
    /// ```text
    /// |S⁽ᵖ⁺¹⁾(ξ)|/(p+1)!  =  |s⁽ᵖ⁾(ξ)|/(p+1)!
    ///                    ≤  p!·sup|s|/(r^p·(p+1)!)
    ///                    =  cosh(π(M+r)r) / ((p+1)·r^p),
    /// ```
    ///
    /// with `M = sup|ξ|` over the argument enclosure.  Every candidate `r`
    /// below is therefore a *valid* bound and the choice among them is purely
    /// a tightness question — the smallest is kept, and a minimum of valid
    /// upper bounds is a valid upper bound.  The same estimate at the
    /// expansion point alone, `|aₖ| ≤ cosh(π(|m₀|+r)r)·r/rᵏ`, feeds
    /// [`Self::geometric_tail`], which is usually far tighter because it never
    /// takes a supremum over the whole box.
    ///
    /// The bound degrades as `x` grows — `S′` oscillates with instantaneous
    /// frequency `πx`, so a box wider than `1/x` genuinely cannot be modelled
    /// at low order and `bound_on_box` must subdivide.  That is a property of
    /// the function, not a weakness of the estimate.
    fn fresnel(&self, sine: bool) -> Result<Self> {
        self.check_finite("Fresnel argument")?;
        let prec = self.prec;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "Fresnel argument".into(),
            });
        }
        let c0 = from_float(&m0, prec);
        let p1 = self.order + 1;

        let (s_at, c_at) = crate::primitive::fresnel::fresnel_pair_ball(&m0, prec).ok_or(
            ValidatedError::NotFinite {
                what: "Fresnel expansion point".into(),
            },
        )?;
        let mut a = Vec::with_capacity(p1);
        a.push(if sine { s_at } else { c_at });
        if self.order >= 1 {
            let (sd, cd) = Self::fresnel_phase_derivs(&c0, self.order, prec);
            let src = if sine { &sd } else { &cd };
            for k in 1..=self.order {
                a.push(Self::div_ball(&src[k - 1], &Self::factorial(k, prec))?);
            }
        }

        // Candidate Cauchy radii: `p/(π·M)` is where the bound is minimised
        // for large `M`, and a fixed ladder covers small `M`, where the
        // optimum runs off to infinity.
        let m_sup = mag(&arg);
        let m_f64 = m_sup.to_f64().max(0.0);
        let mut radii: Vec<f64> = vec![0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
        if m_f64 > 0.0 && self.order >= 1 {
            radii.push((self.order as f64) / (std::f64::consts::PI * m_f64));
        }
        let dpow = ArbBall {
            mid: delta.delta_pow(&d),
            rad: Float::new(prec),
            prec,
        };
        let mut lagrange: Option<Float> = None;
        let mut tail: Option<Float> = None;
        for r in radii {
            if !(r.is_finite() && r > 0.0) {
                continue;
            }
            let rf = Float::with_val(prec, r);
            let rb = from_float(&rf, prec);
            let denom = rb.powi(self.order as i64) * ArbBall::from_f64(p1 as f64, prec);
            let sup = Self::fresnel_phase_bound(&m_sup, &rf, prec);
            if let Ok(q) = Self::div_ball(&sup, &denom) {
                let cand = ub(&(q * dpow.clone()));
                keep_smaller(&mut lagrange, cand);
            }
            let sup_c = Self::fresnel_phase_bound(&Float::with_val(prec, m0.abs_ref()), &rf, prec);
            if let Some(t) = Self::geometric_tail(&(sup_c * rb), &mag(&d), &rf, p1, prec) {
                keep_smaller(&mut tail, t);
            }
        }
        let radius = Self::tighter(
            lagrange.ok_or_else(|| ValidatedError::NotFinite {
                what: "Fresnel remainder bound".into(),
            })?,
            tail,
        );
        let out = delta.compose(&a, &radius);
        out.check_finite("Fresnel result")?;
        Ok(out)
    }

    /// `S(self) = ∫₀^self sin(πt²/2) dt`.
    pub fn fresnel_s(&self) -> Result<Self> {
        self.fresnel(true)
    }

    /// `C(self) = ∫₀^self cos(πt²/2) dt`.
    pub fn fresnel_c(&self) -> Result<Self> {
        self.fresnel(false)
    }

    // ── Dilogarithm (3.10.0) ─────────────────────────────────────────────

    /// Upper bound on `|w(z)| = |−log(1−z)/z|` — the derivative of `Li₂` —
    /// over every `z` within `r` of the real interval `[lo, hi]`.
    ///
    /// `None` unless `1 − hi − r > 0`, i.e. unless the whole disc stays clear
    /// of the branch cut `[1, ∞)`.
    ///
    /// Split at `|z| = 1/2`, which is what makes this uniform where the naive
    /// quotient bound is not — `w` is analytic at the origin, so a bound that
    /// divides by `|z|` falls apart on any disc containing `0` even though
    /// nothing is wrong there:
    ///
    /// * `|z| ≤ 1/2`: `|w(z)| ≤ Σ|z|ᵏ/(k+1) = −log(1−|z|)/|z| ≤ 2·log 2`,
    ///   the majorant being increasing in `|z|`;
    /// * `|z| > 1/2`: `|w(z)| ≤ |log(1−z)|/|z| ≤ 2·(|log|1−z|| + π)`, using
    ///   `|log ζ| ≤ |log|ζ|| + |arg ζ|` on the principal branch.
    ///
    /// and `|1 − z| ∈ [1 − hi − r, 1 − lo + r]` for `z` in the region.
    fn dilog_w_bound(lo: &Float, hi: &Float, r: &Float, prec: u32) -> Option<ArbBall> {
        let rb = from_float(r, prec);
        let one = ArbBall::from_f64(1.0, prec);
        let near = lb(&(one.clone() - from_float(hi, prec) - rb.clone()));
        if !strictly_positive(&near) {
            return None;
        }
        let far = ub(&(one - from_float(lo, prec) + rb));
        let l_near = from_float(&near, prec).log()?;
        let l_far = from_float(&far, prec).log()?;
        let biggest = if mag(&l_near) > mag(&l_far) {
            mag(&l_near)
        } else {
            mag(&l_far)
        };
        let pi = Self::pi_ball(prec);
        let quotient = (from_float(&biggest, prec) + pi) * ArbBall::from_f64(2.0, prec);
        // 2·log 2 ≈ 1.3863, the `|z| ≤ 1/2` branch.
        let series =
            ArbBall::from_f64(1.386_294_361_119_891, prec) + ArbBall::from_f64(1e-15, prec);
        let out = if mag(&quotient) > mag(&series) {
            quotient
        } else {
            series
        };
        is_finite(&out).then_some(out)
    }

    /// Below this expansion point the coefficient recurrence is run
    /// **backwards**; at or above it, forwards.  See
    /// [`Self::dilog_deriv_coeffs`].
    const DILOG_FORWARD_FROM: f64 = 0.4;

    /// Taylor coefficients `uⱼ = w⁽ʲ⁾(m₀)/j!` of `w = Li₂′` for `j = 0..count`.
    ///
    /// From `x·w(x) = −log(1−x)`, expanding both sides about `m₀` with
    /// `v₀ = −log(1−m₀)` and `vⱼ = 1/(j·(1−m₀)ʲ)` gives
    ///
    /// ```text
    /// m₀·u₀ = v₀,        m₀·uⱼ + u_{j−1} = vⱼ   (j ≥ 1).
    /// ```
    ///
    /// **Two directions, because one recurrence is stable in exactly the range
    /// where the other is not.**  The coefficients have the natural size
    /// `|uⱼ| ~ ρ⁻ʲ` with `ρ = 1 − m₀` the distance to the branch point, so:
    ///
    /// * run **forwards** (`uⱼ = (vⱼ − u_{j−1})/m₀`) and a relative error is
    ///   multiplied by `ρ/|m₀|` per step — contracting iff `|m₀| > 1 − m₀`;
    /// * run **backwards** (`u_{j−1} = vⱼ − m₀·uⱼ`, Miller's algorithm) and it
    ///   is multiplied by `|m₀|/ρ` — contracting iff `|m₀| < 1 − m₀`.
    ///
    /// The crossover is `m₀ = 1/2`; the switch is set slightly below it, at
    /// [`Self::DILOG_FORWARD_FROM`] `= 0.4`, so that *both* branches run with a
    /// contraction factor of at most `2/3` rather than one of them sitting at
    /// exactly `1`.  Backwards needs no accurate starting value — only a
    /// rigorous *bound* on `u_J`, from Cauchy's estimate — and the interval
    /// then contracts on the way down, so what comes out is sound whatever `J`
    /// is; `J` only decides how tight.
    ///
    /// `J` is chosen so the contraction has eaten `prec` bits, capped at
    /// `count + 2048`.  Beyond that cap — which needs `m₀ ≲ −180` at 96 bits —
    /// the coefficients stay *rigorous* but widen, and `bound_on_box` returns a
    /// looser (never a wrong) enclosure.
    ///
    /// **The Cauchy radius has to be pressed right up against `ρ`.**  Absolute
    /// error grows by `|m₀|` per backward step, so a start built at radius `r`
    /// arrives at `j = 0` multiplied by `(|m₀|/r)^J`; taking a comfortable
    /// `r = 0.9ρ` instead of `r ≈ ρ` therefore costs a factor `(1/0.9)^J`,
    /// which at `J ≈ 1100` is `e¹¹⁸` and destroys the answer.  `r = ρ(1 −
    /// 1/2J)` costs only `(1 − 1/2J)^{−J} → √e ≈ 1.65`, and `M(r)` — the
    /// numerator — grows merely like `log(2J)` as `r → ρ`, because `w` blows
    /// up only logarithmically at the branch point.
    fn dilog_deriv_coeffs(m0: &ArbBall, count: usize, prec: u32) -> Result<Vec<ArbBall>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let one = ArbBall::from_f64(1.0, prec);
        let one_minus = one.clone() - m0.clone();
        if !strictly_positive(&lb(&one_minus)) {
            return Err(ValidatedError::DomainViolation {
                what: "dilog expansion point at or past the branch point 1".into(),
            });
        }
        let q = Self::div_ball(&one, &one_minus)?;
        let v0 = -one_minus
            .clone()
            .log()
            .ok_or_else(|| ValidatedError::DomainViolation {
                what: "log of a non-positive interval in the dilog expansion".into(),
            })?;
        let m0f = m0.mid.to_f64();

        if m0f >= Self::DILOG_FORWARD_FROM {
            let mut out = Vec::with_capacity(count);
            let mut u = Self::div_ball(&v0, m0)?;
            out.push(u.clone());
            let mut pw = q.clone(); // qʲ
            for j in 1..count {
                let vj = Self::div_ball(&pw, &ArbBall::from_f64(j as f64, prec))?;
                u = Self::div_ball(&(vj - u), m0)?;
                out.push(u.clone());
                pw = pw * q.clone();
            }
            return Ok(out);
        }
        let mut out = vec![ArbBall::from_f64(0.0, prec); count];

        let ratio = (1.0 - m0f) / m0f.abs();
        let steps = if ratio.is_finite() && ratio > 1.0 {
            let s = f64::from(prec) * std::f64::consts::LN_2 / ratio.ln();
            if s.is_finite() {
                (s.ceil() as usize).clamp(32, 2048)
            } else {
                32
            }
        } else {
            32
        };
        let jmax = count + steps;

        // r = ρ·(1 − 1/2J): as close to the disc of analyticity as the
        // logarithmic growth of `M(r)` allows.
        let shrink = ArbBall::from_f64(1.0 - 0.5 / (jmax as f64), prec);
        let rho = lb(&(one_minus.clone() * shrink));
        if !strictly_positive(&rho) {
            return Err(ValidatedError::DomainViolation {
                what: "dilog expansion point too close to the branch cut".into(),
            });
        }
        let m0_mid = Float::with_val(prec, &m0.mid);
        let bound = Self::dilog_w_bound(&m0_mid, &m0_mid, &rho, prec).ok_or_else(|| {
            ValidatedError::DomainViolation {
                what: "dilog expansion point too close to the branch cut".into(),
            }
        })?;
        let start = Self::div_ball(&bound, &from_float(&rho, prec).powi(jmax as i64))?;
        let mut u = symmetric(&mag(&start), prec);
        // pw = q^j, walked down by multiplying by (1 − m₀) rather than
        // dividing by q, so the exponent never has to be recomputed.
        let mut pw = q.powi(jmax as i64);
        for j in (1..=jmax).rev() {
            let vj = Self::div_ball(&pw, &ArbBall::from_f64(j as f64, prec))?;
            u = vj - m0.clone() * u;
            if j - 1 < count {
                out[j - 1] = u.clone();
            }
            pw = pw * one_minus.clone();
        }
        Ok(out)
    }

    /// `Li₂(self)`, principal branch.  Refuses unless the argument enclosure
    /// stays strictly left of the branch point `1`.
    ///
    /// **Domain.**  The cut is `[1, ∞)` (DLMF §25.12(i)); `Li₂` is real and
    /// analytic on `(−∞, 1)`.  The point `x = 1` itself is in the *function's*
    /// domain (`Li₂(1) = π²/6`) but not in this rule's: `Li₂′ = −log(1−x)/x`
    /// is unbounded there, so no Taylor model of any order exists on a box
    /// touching it.
    ///
    /// **Coefficients.**  `a₀ = Li₂(m₀)` from MPFR's correctly-rounded
    /// `mpfr_li2`; `a_{j+1} = uⱼ/(j+1)` where `uⱼ` are the coefficients of
    /// `Li₂′` from `dilog_deriv_coeffs`, since integrating the
    /// derivative's series term by term divides the `j`-th coefficient by
    /// `j+1`.
    ///
    /// **Remainder.**  `Li₂` is analytic on `ℂ ∖ [1, ∞)`, so Cauchy's estimate
    /// holds for every `r` with `1 − U − r > 0` (`U` the enclosure's upper
    /// end):
    ///
    /// ```text
    /// |Li₂⁽ᵖ⁺¹⁾(ξ)|/(p+1)!  =  |w⁽ᵖ⁾(ξ)|/(p+1)!  ≤  sup|w| / ((p+1)·r^p),
    /// ```
    ///
    /// with `sup|w|` from `dilog_w_bound`.  Every `r` is valid, so the
    /// ladder tried below is a tightness choice only.  The same estimate at
    /// the expansion point feeds `geometric_tail` with `ρ = r`, and
    /// the smaller of the two is kept.
    pub fn dilog(&self) -> Result<Self> {
        self.check_finite("dilog argument")?;
        let prec = self.prec;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, prec) + d.clone();
        if !is_finite(&arg) {
            return Err(ValidatedError::NotFinite {
                what: "dilog argument".into(),
            });
        }
        let arg_lo = lb(&arg);
        let arg_hi = ub(&arg);
        if !strictly_positive(&Float::with_val(prec, 1.0 - arg_hi.clone())) {
            return Err(ValidatedError::DomainViolation {
                what: "dilog of an argument whose enclosure reaches the branch point at 1 (the cut runs along [1, ∞))".into(),
            });
        }
        let c = from_float(&m0, prec);
        let p1 = self.order + 1;

        let mut a = Vec::with_capacity(p1);
        a.push(
            crate::primitive::polylog::dilog_ball_point(&m0, prec).ok_or_else(|| {
                ValidatedError::DomainViolation {
                    what: "dilog expansion point past the branch point 1".into(),
                }
            })?,
        );
        if self.order >= 1 {
            let u = Self::dilog_deriv_coeffs(&c, self.order, prec)?;
            for (j, uj) in u.iter().enumerate() {
                a.push(Self::div_ball(
                    uj,
                    &ArbBall::from_f64((j + 1) as f64, prec),
                )?);
            }
        }

        let gap_box = Float::with_val(prec, 1.0 - arg_hi.clone());
        let gap_pt = ub(&(ArbBall::from_f64(1.0, prec) - c.clone()));
        let dpow = ArbBall {
            mid: delta.delta_pow(&d),
            rad: Float::new(prec),
            prec,
        };
        let mut lagrange: Option<Float> = None;
        let mut tail: Option<Float> = None;
        for frac in [0.5_f64, 0.75, 0.9, 0.99] {
            let r_box = Float::with_val(prec, &gap_box * frac);
            if strictly_positive(&r_box) {
                if let Some(sup) = Self::dilog_w_bound(&arg_lo, &arg_hi, &r_box, prec) {
                    let denom = from_float(&r_box, prec).powi(self.order as i64)
                        * ArbBall::from_f64(p1 as f64, prec);
                    if let Ok(q) = Self::div_ball(&sup, &denom) {
                        let cand = ub(&(q * dpow.clone()));
                        keep_smaller(&mut lagrange, cand);
                    }
                }
            }
            let r_pt = Float::with_val(prec, &gap_pt * frac);
            if !strictly_positive(&r_pt) {
                continue;
            }
            let m0f = Float::with_val(prec, &m0);
            if let Some(sup) = Self::dilog_w_bound(&m0f, &m0f, &r_pt, prec) {
                let cc = sup * from_float(&r_pt, prec);
                if let Some(t) = Self::geometric_tail(&cc, &mag(&d), &r_pt, p1, prec) {
                    keep_smaller(&mut tail, t);
                }
            }
        }
        let radius = Self::tighter(
            lagrange.ok_or_else(|| ValidatedError::NotFinite {
                what: "dilog remainder bound".into(),
            })?,
            tail,
        );
        let out = delta.compose(&a, &radius);
        out.check_finite("dilog result")?;
        Ok(out)
    }

    // ── Trigamma (3.10.0) ────────────────────────────────────────────────

    /// `ψ₁(self) = ψ′(self)`.  Refuses unless the argument enclosure lies
    /// strictly inside `(0, ∞)`, for the same reason
    /// [`TaylorModel::digamma`] does: the Hurwitz zeta below needs a strictly
    /// positive second argument, and between the negative poles nobody has
    /// written the reflection.
    ///
    /// **Coefficients.**  `ψ₁ = ζ(2, x)` and `∂ₓ^k ζ(s, x) = (−1)ᵏ(s)ₖ ζ(s+k, x)`,
    /// so `ψ₁⁽ᵏ⁾(x) = (−1)ᵏ(k+1)!·ζ(k+2, x)` and
    ///
    /// ```text
    /// aₖ = ψ₁⁽ᵏ⁾(m₀)/k! = (−1)ᵏ·(k+1)·ζ(k+2, m₀),
    /// ```
    ///
    /// exactly, with no recurrence and no cancellation — the same Hurwitz
    /// zetas [`TaylorModel::digamma`] already computes, read one index along.
    ///
    /// **Remainder.**  `|ψ₁⁽ᵖ⁺¹⁾(ξ)|/(p+1)! = (p+2)·ζ(p+3, ξ)`, decreasing in
    /// `ξ` term by term, so its supremum sits at the lower endpoint `L`, where
    /// the integral comparison `ζ(s, L) ≤ L^{−s} + L^{1−s}/(s−1)` applies.
    pub fn trigamma(&self) -> Result<Self> {
        self.check_finite("trigamma argument")?;
        let prec = self.prec;
        let (m0, delta) = self.center_split();
        let d = delta.range();
        let arg = from_float(&m0, prec) + d.clone();
        let arg_lo = lb(&arg);
        if !strictly_positive(&arg_lo) {
            return Err(ValidatedError::DomainViolation {
                what: "trigamma of an argument whose enclosure reaches 0 or below (double poles sit at every non-positive integer)".into(),
            });
        }
        let c = from_float(&m0, prec);
        let p1 = self.order + 1;
        let zetas = Self::hurwitz_zeta_ints(&c, self.order + 2, prec)?;
        let a: Vec<ArbBall> = zetas
            .iter()
            .take(p1)
            .enumerate()
            .map(|(k, zk)| {
                let z = zk.clone() * ArbBall::from_f64((k + 1) as f64, prec);
                if k % 2 == 0 {
                    z
                } else {
                    -z
                }
            })
            .collect();

        // ζ(p+3, L) ≤ L^{-(p+3)} + L^{-(p+2)}/(p+2), times (p+2).
        let lo_ball = from_float(&arg_lo, prec);
        let one = ArbBall::from_f64(1.0, prec);
        let p2 = ArbBall::from_f64((p1 + 1) as f64, prec);
        let sup = (Self::div_ball(&one, &lo_ball.powi((p1 + 2) as i64))?
            + Self::div_ball(&one, &(lo_ball.powi((p1 + 1) as i64) * p2.clone()))?)
            * p2;
        let radius = ub(&(sup
            * ArbBall {
                mid: delta.delta_pow(&d),
                rad: Float::new(prec),
                prec,
            }));
        let out = delta.compose(&a, &radius);
        out.check_finite("trigamma result")?;
        Ok(out)
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
                    "bessel_j0" => x.bessel_j(0),
                    "bessel_j1" => x.bessel_j(1),
                    "digamma" => x.digamma(),
                    "trigamma" => x.trigamma(),
                    "gamma" => x.gamma(),
                    "lambert_w" => x.lambert_w(),
                    // Exponential-integral family. The rules live beside
                    // the primitives in `primitive::expint`; only the
                    // dispatch is here.
                    "Ei" => crate::primitive::expint::taylor_ei(&x),
                    "li" => crate::primitive::expint::taylor_li(&x),
                    "Si" => crate::primitive::expint::taylor_si(&x),
                    "Ci" => crate::primitive::expint::taylor_ci(&x),
                    "Shi" => crate::primitive::expint::taylor_shi(&x),
                    "Chi" => crate::primitive::expint::taylor_chi(&x),
                    "fresnels" => x.fresnel_s(),
                    "fresnelc" => x.fresnel_c(),
                    "dilog" => x.dilog(),
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

    // ── Bessel / digamma / gamma / Lambert W ─────────────────────────────

    /// A deterministic LCG.  The randomised sweeps want *breadth* of box
    /// shapes, and a fixed seed reproduces a failure exactly.
    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// The true value at `t`, at `P + 64` bits — far beyond the enclosure's
    /// own radius, so a containment failure here is the enclosure's fault and
    /// not the reference's.
    ///
    /// `lambert_w` is deliberately absent: MPFR has no `W`, and re-running the
    /// same Newton iteration the kernel uses would test nothing.  It is
    /// checked instead through its defining equation — see
    /// `lambert_w_encloses_by_its_defining_equation`.
    fn ref_special(name: &str, t: &Float) -> Float {
        let v = Float::with_val(P + 64, t);
        match name {
            "digamma" => {
                let mut w = v;
                w.digamma_mut();
                w
            }
            "gamma" => v.gamma(),
            "bessel_j0" => v.jn(0),
            "bessel_j1" => v.jn(1),
            _ => unreachable!("no reference for `{name}`"),
        }
    }

    fn assert_special_encloses(name: &str, r: &ArbBall, t: &Float, ctx: &str) {
        let truth = ref_special(name, t);
        assert!(
            r.lo() <= truth && truth <= r.hi(),
            "{name}({t}) = {truth} escaped [{}, {}] {ctx}",
            r.lo(),
            r.hi()
        );
    }

    /// Containment over hand-picked boxes, including ones that run right up to
    /// a domain boundary (`digamma`/`gamma` towards the pole at 0) and ones
    /// spanning several oscillations of `J₀`/`J₁`, where an endpoint-hull
    /// argument would fail outright.
    #[test]
    fn special_rules_enclose_dense_samples() {
        let cases: [(&str, &[(f64, f64)]); 4] = [
            (
                "bessel_j0",
                &[
                    (-1.0, 1.0),
                    (0.0, 0.0),
                    (2.0, 3.0),
                    (-6.0, 6.0),
                    (10.0, 12.0),
                    (-20.0, -19.5),
                    (2.404, 2.405), // straddles the first zero of J₀
                ],
            ),
            (
                "bessel_j1",
                &[
                    (-1.0, 1.0),
                    (0.0, 0.5),
                    (3.8, 3.84), // straddles the first positive zero of J₁
                    (-5.0, 5.0),
                    (15.0, 16.0),
                ],
            ),
            (
                "digamma",
                &[
                    (1.0, 2.0),
                    (0.25, 0.5),
                    (0.001, 0.0011),
                    (5.0, 9.0),
                    (100.0, 101.0),
                    (1.4, 1.5), // straddles the zero of ψ
                ],
            ),
            (
                "gamma",
                &[
                    (1.0, 2.0),
                    (0.5, 0.75),
                    (0.01, 0.02),
                    (1.4, 1.5), // straddles the minimum of Γ
                    (3.0, 4.0),
                    (7.0, 7.5),
                ],
            ),
        ];
        for (name, boxes) in cases {
            for &(lo, hi) in boxes {
                let r = tm_range(name, lo, hi, 8)
                    .unwrap_or_else(|e| panic!("{name} on [{lo},{hi}]: {e}"));
                for t in samples(lo, hi, 200) {
                    assert_special_encloses(name, &r, &t, &format!("on [{lo},{hi}]"));
                }
            }
        }
    }

    /// Degenerate boxes: pin the *value*, which no containment sweep over a
    /// wide box can guarantee to catch.  The radius bound also pins that the
    /// Hurwitz-zeta coefficients really are computed to working precision
    /// rather than to whatever the Euler–Maclaurin truncation happened to give.
    #[test]
    fn special_point_values_are_pinned() {
        let cases: [(&str, &[f64]); 4] = [
            ("bessel_j0", &[0.0, 1.0, -1.0, 2.5, 7.25, -13.5]),
            ("bessel_j1", &[0.0, 1.0, -2.0, 4.75, 11.0]),
            ("digamma", &[0.5, 1.0, 2.0, 3.75, 40.0, 0.01]),
            ("gamma", &[0.5, 1.0, 1.4616, 2.0, 6.5, 0.02]),
        ];
        for (name, points) in cases {
            for &p in points {
                let r = tm_range(name, p, p, 6).unwrap_or_else(|e| panic!("{name}({p}): {e}"));
                let t = Float::with_val(P, p);
                assert_special_encloses(name, &r, &t, "at a degenerate box");
                let rel = r.rad_f64() / (1.0 + r.mid_f64().abs());
                assert!(
                    rel < 1e-25,
                    "{name}({p}) should be a point evaluation, relative radius {rel}"
                );
            }
        }
    }

    /// `W₀` has no MPFR reference, so containment is checked through the
    /// equation that *defines* it: `g(w) = w·eʷ` is strictly increasing on
    /// `w > −1`, so `W₀(t) ∈ [lo, hi]` **iff** `g(lo) ≤ t ≤ g(hi)`.  That uses
    /// nothing but `exp`, and in particular no part of the code under test.
    #[test]
    fn lambert_w_encloses_by_its_defining_equation() {
        // `W₀(t) ∈ [lo, hi]` ⟺ `g(lo) ≤ t ≤ g(hi)`, but only for `lo, hi ≥ −1`
        // where `g` is increasing.  An enclosure reaching below `−1` covers
        // the whole branch on that side and needs no check; one whose *upper*
        // end is below `−1` is a genuine escape, since `W₀ ≥ −1` always.
        let brackets = |r: &ArbBall, t: &Float| -> bool {
            let g = |w: &Float| -> Float {
                let e = Float::with_val(P + 64, w).exp();
                Float::with_val(P + 64, w) * e
            };
            let minus_one = Float::with_val(P + 64, -1);
            let below = r.lo() <= minus_one || g(&r.lo()) <= *t;
            let above = r.hi() >= minus_one && g(&r.hi()) >= *t;
            below && above
        };
        for &(lo, hi) in &[
            (0.0_f64, 1.0_f64),
            (-0.3, 0.0),
            (1.0, 2.0),
            (-0.36, -0.35),
            (10.0, 20.0),
            (1e4, 1e5),
            (0.5, 0.5),
            (-0.2, 0.8),
        ] {
            let r = tm_range("lambert_w", lo, hi, 8)
                .unwrap_or_else(|e| panic!("lambert_w on [{lo},{hi}]: {e}"));
            for t in samples(lo, hi, 200) {
                assert!(
                    brackets(&r, &t),
                    "W₀({t}) escaped [{}, {}] on [{lo},{hi}]",
                    r.lo(),
                    r.hi()
                );
            }
        }
    }

    /// Off-domain and boundary-touching boxes refuse with a *domain*
    /// violation, not with "no such rule".
    #[test]
    fn special_rules_refuse_off_domain_boxes() {
        for (name, lo, hi) in [
            // digamma / gamma: poles at 0, −1, −2, …
            ("digamma", 0.0, 1.0),
            ("digamma", -0.5, 0.5),
            ("digamma", -3.0, -2.0), // between poles, but still refused
            ("digamma", -1.0, -1.0),
            ("gamma", 0.0, 1.0),
            ("gamma", -0.5, 0.5),
            ("gamma", -2.5, -2.4),
            ("gamma", -4.0, -1.0),
            // lambert_w: the principal branch starts at −1/e ≈ −0.36788.
            ("lambert_w", -1.0, 1.0),
            ("lambert_w", -0.5, -0.4),
            ("lambert_w", -0.4, 0.0),
            ("lambert_w", -1e6, -1e5),
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

    /// `J₀`/`J₁` are entire: no box may refuse on domain grounds, including
    /// wide boxes spanning many oscillations and boxes centred on a zero.
    #[test]
    fn bessel_never_refuses_on_domain_grounds() {
        for name in ["bessel_j0", "bessel_j1"] {
            for (lo, hi) in [
                (-1.0, 1.0),
                (0.0, 0.0),
                (-40.0, 40.0),
                (2.404_825, 2.404_825),
                (-100.0, -99.0),
            ] {
                let r = tm_range(name, lo, hi, 6)
                    .unwrap_or_else(|e| panic!("{name} on [{lo},{hi}] refused: {e}"));
                for t in samples(lo, hi, 30) {
                    assert_special_encloses(name, &r, &t, &format!("on [{lo},{hi}]"));
                }
            }
        }
    }

    /// The enclosure a *hull* would have produced is excluded explicitly: on
    /// `[-1, 1]` the endpoints of `J₀` agree at 0.7651977 and the maximum
    /// `J₀(0) = 1` sits strictly inside.  This is the exact configuration that
    /// made the ball kernel unsound in 3.8.
    #[test]
    fn bessel_covers_the_interior_maximum_a_hull_would_miss() {
        let r = tm_range("bessel_j0", -1.0, 1.0, 10).unwrap();
        assert!(
            r.hi() >= 1.0,
            "J₀(0) = 1 must be enclosed by the bound over [-1,1], got {r}"
        );
        assert!(r.lo() <= 0.7651, "the endpoint value escaped: {r}");
        // …and the bound is not merely true: J₀ ranges over [0.7652, 1] there.
        assert!(width_of(&r) < 0.5, "enclosure width {}", width_of(&r));
    }

    /// The enclosures have to be usable, not merely true: measured against the
    /// width of the true range on the box.
    #[test]
    fn special_enclosures_are_tight() {
        for (name, lo, hi) in [
            ("bessel_j0", 2.0, 2.5),
            ("bessel_j1", 1.0, 1.5),
            ("digamma", 1.0, 1.5),
            ("digamma", 4.0, 5.0),
            ("gamma", 1.0, 1.5),
            ("gamma", 3.0, 3.5),
        ] {
            let r = tm_range(name, lo, hi, 12).unwrap();
            let mut span = f64::NEG_INFINITY;
            let mut lowest = f64::INFINITY;
            for t in samples(lo, hi, 64) {
                let v = ref_special(name, &t).to_f64();
                span = span.max(v);
                lowest = lowest.min(v);
            }
            let true_width = span - lowest;
            assert!(
                width_of(&r) <= 2.0 * true_width + 1e-12,
                "{name} on [{lo},{hi}]: enclosure width {} against a true range of {true_width}",
                width_of(&r)
            );
        }
    }

    /// Randomised sweep: 200 boxes per function, each checked for containment
    /// of 40 densely sampled true values.  Every escape is a soundness bug.
    #[test]
    fn special_rules_random_box_sweep() {
        let mut next = lcg(0x51ED_2701_C0FF);
        for name in ["bessel_j0", "bessel_j1", "digamma", "gamma"] {
            let mut checked = 0usize;
            for _ in 0..200 {
                let (lo, hi) = match name {
                    // Entire: anywhere on ℝ, widths from a point to wide.
                    "bessel_j0" | "bessel_j1" => {
                        let c = (next() - 0.5) * 40.0;
                        let w = next().powi(3) * 3.0;
                        (c - w, c + w)
                    }
                    // Poles at 0, −1, …: strictly positive by construction,
                    // with the pole at the origin approached but never met.
                    "digamma" => {
                        let lo = next().powi(4) * 20.0 + 1e-3;
                        let w = next().powi(3) * 2.0;
                        (lo, lo + w)
                    }
                    _ => {
                        let lo = next().powi(4) * 8.0 + 1e-2;
                        let w = next().powi(3) * 1.5;
                        (lo, lo + w)
                    }
                };
                let r = match tm_range(name, lo, hi, 8) {
                    Ok(r) => r,
                    // A refusal is always sound; the refusal *reasons* are
                    // pinned by `special_rules_refuse_off_domain_boxes`.
                    Err(_) => continue,
                };
                checked += 1;
                for t in samples(lo, hi, 40) {
                    assert_special_encloses(name, &r, &t, &format!("on [{lo}, {hi}]"));
                }
            }
            assert!(
                checked > 100,
                "{name}: only {checked}/200 boxes produced a bound — the sweep is not exercising the rule"
            );
        }
    }

    /// The same sweep for `W₀`, through its defining equation.
    #[test]
    fn lambert_w_random_box_sweep() {
        let mut next = lcg(0x11A3_B0C7_5E11);
        // `W₀(t) ∈ [lo, hi]` ⟺ `g(lo) ≤ t ≤ g(hi)`, but only for `lo, hi ≥ −1`
        // where `g` is increasing.  An enclosure reaching below `−1` covers
        // the whole branch on that side and needs no check; one whose *upper*
        // end is below `−1` is a genuine escape, since `W₀ ≥ −1` always.
        let brackets = |r: &ArbBall, t: &Float| -> bool {
            let g = |w: &Float| -> Float {
                let e = Float::with_val(P + 64, w).exp();
                Float::with_val(P + 64, w) * e
            };
            let minus_one = Float::with_val(P + 64, -1);
            let below = r.lo() <= minus_one || g(&r.lo()) <= *t;
            let above = r.hi() >= minus_one && g(&r.hi()) >= *t;
            below && above
        };
        let mut checked = 0usize;
        for _ in 0..200 {
            // Strictly right of −1/e by construction; the offset is drawn on a
            // quartic so most boxes crowd the branch point, which is where a
            // remainder bound is hardest.
            let lo = -0.367_879_441_171_442 + next().powi(4) * 30.0 + 1e-4;
            let w = next().powi(3) * (lo + 0.367_879_441_171_442).min(2.0);
            let (lo, hi) = (lo, lo + w);
            let r = match tm_range("lambert_w", lo, hi, 8) {
                Ok(r) => r,
                Err(_) => continue,
            };
            checked += 1;
            for t in samples(lo, hi, 40) {
                assert!(
                    brackets(&r, &t),
                    "W₀({t}) escaped [{}, {}] on [{lo}, {hi}]",
                    r.lo(),
                    r.hi()
                );
            }
        }
        assert!(checked > 100, "only {checked}/200 boxes produced a bound");
    }

    /// The Hurwitz zeta the `digamma`/`gamma` coefficients are built from,
    /// against two independent references: MPFR's Riemann `ζ(s)` at `a = 1`,
    /// and the functional equation `ζ(s,a) − ζ(s,a+1) = a^{-s}` at arbitrary
    /// `a`.  A sign error in the Euler–Maclaurin corrections cannot survive
    /// either — the second in particular compares two evaluations whose
    /// correction terms differ.
    #[test]
    fn hurwitz_zeta_matches_independent_references() {
        let prec = 160u32;
        let s_max = 26usize;
        let one = ArbBall::from_f64(1.0, prec);
        let at_one = TaylorModel::hurwitz_zeta_ints(&one, s_max, prec).unwrap();
        for (i, z) in at_one.iter().enumerate() {
            let s = i + 2;
            let truth = Float::with_val(prec + 64, s as u32).zeta();
            assert!(
                z.lo() <= truth && truth <= z.hi(),
                "ζ({s}) = {truth} escaped [{}, {}]",
                z.lo(),
                z.hi()
            );
            assert!(z.rad_f64() < 1e-40, "ζ({s}) radius {}", z.rad_f64());
        }

        for a in [0.25_f64, 0.5, 1.5, 3.0, 7.75, 60.0] {
            let ab = ArbBall::from_f64(a, prec);
            let bb = ArbBall::from_f64(a + 1.0, prec);
            let za = TaylorModel::hurwitz_zeta_ints(&ab, s_max, prec).unwrap();
            let zb = TaylorModel::hurwitz_zeta_ints(&bb, s_max, prec).unwrap();
            for (i, (x, y)) in za.iter().zip(&zb).enumerate() {
                let s = i + 2;
                let diff = x.clone() - y.clone();
                let expect = Float::with_val(
                    prec + 64,
                    Float::with_val(prec + 64, 1)
                        / rug::ops::Pow::pow(Float::with_val(prec + 64, a), s as u32),
                );
                assert!(
                    diff.lo() <= expect && expect <= diff.hi(),
                    "ζ({s},{a}) − ζ({s},{}) should be {expect}, got [{}, {}]",
                    a + 1.0,
                    diff.lo(),
                    diff.hi()
                );
            }
        }
    }

    /// The identities that tie the new rules to the rest of the algebra.
    /// These sit *next to* the containment tests, never instead of them: a
    /// functional equation is invariant under exactly the kind of sign flip
    /// that a containment check catches.
    #[test]
    fn special_rules_satisfy_their_functional_equations() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        // Γ(x+1) = x·Γ(x)
        let shifted = pool.func("gamma", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        let scaled = pool.mul(vec![x, pool.func("gamma", vec![x])]);
        let diff = sub(&pool, shifted, scaled);
        let r = range_of(diff, &pool, &[(x, 1.0, 2.0)], 12);
        assert!(r.contains(0.0), "Γ(x+1) − xΓ(x) should vanish, got {r:?}");
        assert!(r.rad_f64() < 1e-3, "…and tightly: {r:?}");

        // ψ(x+1) = ψ(x) + 1/x
        let lhs = pool.func("digamma", vec![pool.add(vec![x, pool.integer(1_i32)])]);
        let rhs = pool.add(vec![
            pool.func("digamma", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        let r = range_of(sub(&pool, lhs, rhs), &pool, &[(x, 1.0, 2.0)], 12);
        assert!(r.contains(0.0), "ψ(x+1) − ψ(x) − 1/x: {r:?}");
        assert!(r.rad_f64() < 1e-3, "…and tightly: {r:?}");

        // W(x)·e^{W(x)} = x
        let w = pool.func("lambert_w", vec![x]);
        let e = pool.mul(vec![w, pool.func("exp", vec![w])]);
        let r = range_of(sub(&pool, e, x), &pool, &[(x, 0.5, 1.5)], 12);
        assert!(r.contains(0.0), "W·e^W − x: {r:?}");
        assert!(r.rad_f64() < 1e-2, "…and tightly: {r:?}");

        // J₀′ = −J₁, checked as J₀(x)² + J₁(x)² ≤ 1 (Bessel's own bound) plus
        // the ODE x·J₀″ + J₀′ + x·J₀ = 0 is not expressible here; instead pin
        // the recurrence-free fact that both stay inside [−1, 1].
        for name in ["bessel_j0", "bessel_j1"] {
            let r = range_of(pool.func(name, vec![x]), &pool, &[(x, -30.0, 30.0)], 4);
            assert!(
                r.lo() <= 1.0 && r.hi() >= -1.0,
                "{name} enclosure {r:?} is inconsistent with |J| ≤ 1"
            );
        }
    }

    /// The rules compose with the rest of the algebra and over several
    /// variables, which is what makes them Taylor models rather than point
    /// evaluators.
    #[test]
    fn special_rules_compose() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // J₀(x² + y) on [0,1]×[1,2]
        let arg = pool.add(vec![pool.mul(vec![x, x]), y]);
        let r = range_of(
            pool.func("bessel_j0", vec![arg]),
            &pool,
            &[(x, 0.0, 1.0), (y, 1.0, 2.0)],
            8,
        );
        for i in 0..=20 {
            for j in 0..=20 {
                let a = Float::with_val(P + 64, i as f64 / 20.0);
                let b = Float::with_val(P + 64, 1.0 + j as f64 / 20.0);
                let truth = (a.clone() * a + b).jn(0);
                assert!(
                    r.lo() <= truth && truth <= r.hi(),
                    "J₀(x²+y) escaped {r:?} at ({i},{j})"
                );
            }
        }
        // Γ(x)·ψ(x) on [1.5, 2.5]
        let e = pool.mul(vec![
            pool.func("gamma", vec![x]),
            pool.func("digamma", vec![x]),
        ]);
        let r = range_of(e, &pool, &[(x, 1.5, 2.5)], 10);
        for t in samples(1.5, 2.5, 50) {
            let mut psi = Float::with_val(P + 64, &t);
            psi.digamma_mut();
            let truth = Float::with_val(P + 64, &t).gamma() * psi;
            assert!(
                r.lo() <= truth && truth <= r.hi(),
                "Γψ({t}) = {truth} escaped {r:?}"
            );
        }
    }

    /// Order and precision are both free parameters of the enclosure; sweep
    /// them, because the remainder scales in the first and the coefficient
    /// accuracy in the second, and a bound that is only correct at the default
    /// settings is not a bound.
    #[test]
    fn special_rules_hold_across_orders_and_precisions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut checked = 0usize;
        for name in ["bessel_j0", "bessel_j1", "digamma", "gamma"] {
            for &(lo, hi) in &[(1.25_f64, 1.75_f64), (0.5, 0.75), (3.0, 3.5)] {
                for order in [1usize, 2, 5, 11, 24] {
                    for prec in [32u32, 64, 128, 256] {
                        let e = pool.func(name, vec![x]);
                        let boxes = vec![(x, Float::with_val(prec, lo), Float::with_val(prec, hi))];
                        let Ok(r) = taylor_range(e, &pool, &boxes, order, prec) else {
                            continue;
                        };
                        checked += 1;
                        for t in samples(lo, hi, 12) {
                            assert_special_encloses(
                                name,
                                &r,
                                &t,
                                &format!("on [{lo},{hi}] at order {order}, prec {prec}"),
                            );
                        }
                    }
                }
            }
        }
        assert!(
            checked > 100,
            "only {checked} configurations produced a bound"
        );
    }

    #[test]
    fn unsupported_function_refuses() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let e = pool.func("EllipticK", vec![x]);
        let boxes = vec![(x, f(0.1), f(0.2))];
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

    // ── 3.10.0 rules ─────────────────────────────────────────────────────

    /// A rule that produces a *finite* enclosure is worth nothing; the point
    /// of this tier is that the enclosure **contains** the value.  Each of the
    /// three checks below asks a degenerate box for a published constant.
    fn encloses(cases: &[(&str, f64, f64)]) {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions {
            order: 5,
            prec: 128,
            tol: 1e-12,
            max_subdivisions: 64,
        };
        for &(name, at, want) in cases {
            let e = pool.func(name, vec![x]);
            let r = bound_on_box(e, &pool, &[(x, at, at)], &opts)
                .unwrap_or_else(|err| panic!("{name}({at}): {err}"));
            let lo = lb(r.enclosure()).to_f64();
            let hi = ub(r.enclosure()).to_f64();
            assert!(
                lo - 1e-12 <= want && want <= hi + 1e-12,
                "{name}({at}): {want} outside [{lo}, {hi}]"
            );
        }
    }

    /// `ψ₁(1) = π²/6`, `ψ₁(2) = π²/6 − 1` and `ψ₁(½) = π²/2` — A&S 6.4.2 and
    /// 6.4.4.
    #[test]
    fn the_trigamma_rule_encloses_its_published_values() {
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        encloses(&[
            ("trigamma", 1.0, pi2 / 6.0),
            ("trigamma", 2.0, pi2 / 6.0 - 1.0),
            ("trigamma", 0.5, pi2 / 2.0),
        ]);
    }

    /// `ψ₁` has a double pole at every non-positive integer.  The strips
    /// *between* the negative poles are analytic but are not covered, exactly
    /// as for [`TaylorModel::digamma`].
    #[test]
    fn the_trigamma_rule_refuses_at_and_below_its_poles() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions::default();
        for (lo, hi) in [(-0.5_f64, 0.5_f64), (0.0, 1.0), (-3.0, -2.0)] {
            let e = pool.func("trigamma", vec![x]);
            assert!(
                bound_on_box(e, &pool, &[(x, lo, hi)], &opts).is_err(),
                "trigamma on [{lo}, {hi}] must refuse"
            );
        }
    }

    /// A&S Table 7.7 / `scipy.special.fresnel`, in the normalised π/2
    /// convention.  `x = 8` is past the series/asymptotic seam at `|x| = 6`,
    /// so both regimes of the point kernel are exercised.
    #[test]
    fn the_fresnel_rules_enclose_their_published_values() {
        encloses(&[
            ("fresnels", 1.0, 0.438_259_147_390_354_7),
            ("fresnelc", 1.0, 0.779_893_400_376_823),
            ("fresnels", 3.0, 0.496_312_998_967_375),
            ("fresnelc", 8.0, 0.499_802_180_377_197_15),
        ]);
    }

    /// `S` and `C` are entire, so — unlike every other special-function rule
    /// here — there is no box they may refuse for a domain reason.
    #[test]
    fn the_fresnel_rules_refuse_nothing() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions::default();
        for name in ["fresnels", "fresnelc"] {
            let e = pool.func(name, vec![x]);
            for (lo, hi) in [(-50.0_f64, -49.0_f64), (-0.5, 0.5), (9.0, 9.5)] {
                assert!(
                    bound_on_box(e, &pool, &[(x, lo, hi)], &opts).is_ok(),
                    "{name} on [{lo}, {hi}]"
                );
            }
        }
    }

    /// `Li₂(−1) = −π²/12` (DLMF 25.12.2) and `Li₂(½) = π²/12 − log²2/2`
    /// (Lewin eq. 1.16).
    #[test]
    fn the_dilog_rule_encloses_its_published_values() {
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        let ln2 = std::f64::consts::LN_2;
        encloses(&[
            ("dilog", -1.0, -pi2 / 12.0),
            ("dilog", 0.5, pi2 / 12.0 - 0.5 * ln2 * ln2),
            ("dilog", 0.0, 0.0),
        ]);
    }

    /// Far into the inversion branch the coefficient recurrence runs backwards
    /// for over a thousand steps.  There is no published closed form here, so
    /// the reference is MPFR's correctly-rounded `mpfr_li2`; the enclosure has
    /// to stay tight, not merely finite — an over-wide answer here is exactly
    /// what a mis-chosen Cauchy radius produces.
    #[test]
    fn the_dilog_rule_stays_tight_on_the_far_inversion_branch() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions {
            order: 5,
            prec: 128,
            tol: 1e-12,
            max_subdivisions: 64,
        };
        for at in [-30.0_f64, -200.0] {
            let e = pool.func("dilog", vec![x]);
            let r = bound_on_box(e, &pool, &[(x, at, at)], &opts)
                .unwrap_or_else(|err| panic!("dilog({at}): {err}"));
            let lo = lb(r.enclosure()).to_f64();
            let hi = ub(r.enclosure()).to_f64();
            let want = Float::with_val(200, at).li2().to_f64();
            assert!(hi - lo < 1e-10, "dilog({at}): [{lo}, {hi}] is loose");
            assert!(
                lo - 1e-12 <= want && want <= hi + 1e-12,
                "dilog({at}): {want} outside [{lo}, {hi}]"
            );
        }
    }

    /// The dilog rule has two coefficient recurrences meeting at `m₀ = 0.4`.
    /// They compute the same numbers, so either side of the seam must agree
    /// with the closed forms — a mismatch here is the bug a single-branch test
    /// would never see.
    #[test]
    fn the_dilog_coefficient_recurrences_agree_across_their_seam() {
        let prec = 128;
        for m in [0.399_999_9_f64, 0.4, 0.400_000_1] {
            let c = ArbBall::from_f64(m, prec);
            let u = TaylorModel::dilog_deriv_coeffs(&c, 8, prec).unwrap();
            // u₀ = Li₂′(m) = −log(1−m)/m, independently.
            let want = -(1.0 - m).ln() / m;
            assert!(
                (u[0].mid.to_f64() - want).abs() < 1e-12,
                "m = {m}: u₀ = {} vs {want}",
                u[0].mid.to_f64()
            );
            // u₁ = Li₂″(m) = 1/((1−m)m) + log(1−m)/m².
            let want1 = 1.0 / ((1.0 - m) * m) + (1.0 - m).ln() / (m * m);
            assert!(
                (u[1].mid.to_f64() - want1).abs() < 1e-10,
                "m = {m}: u₁ = {} vs {want1}",
                u[1].mid.to_f64()
            );
            assert!(u.iter().all(is_finite), "m = {m}: non-finite");
        }
    }

    /// The cut is `[1, ∞)`.  `x = 1` is in the *function's* domain
    /// (`Li₂(1) = π²/6`) but not in this rule's: `Li₂′ = −log(1−x)/x` is
    /// unbounded there, so no Taylor model of any order exists on a box
    /// touching it.
    #[test]
    fn the_dilog_rule_refuses_past_its_branch_cut() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions::default();
        for (lo, hi) in [(0.5_f64, 1.0_f64), (1.0, 2.0), (2.0, 3.0)] {
            let e = pool.func("dilog", vec![x]);
            assert!(
                bound_on_box(e, &pool, &[(x, lo, hi)], &opts).is_err(),
                "dilog on [{lo}, {hi}] must refuse"
            );
        }
    }
}
