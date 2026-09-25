//! The value type this module hands back: a **complex ball**, `[mid ± rad]` on
//! each component, carrying enough information for a caller to decide whether
//! it is worth reading.
//!
//! These are deliberately *not* [`crate::ball::AcbBall`]. That type is the
//! MPFR-backed interval evaluator used by `IntervalEval`; [`ComplexBall`] is
//! what comes out of genuine `acb_t` arithmetic inside FLINT. The two are kept
//! apart because conflating them would make it impossible to tell, at a call
//! site, which error model produced a radius.
//!
//! # Reading a result honestly
//!
//! [`ComplexBall::midpoint_re`] is named *midpoint*, not *value*, on purpose.
//! The midpoint of `[0 ± 10^9]` is `0` and means nothing. Three accessors exist
//! to stop that from being read as an answer:
//!
//! * [`ComplexBall::accuracy_bits`] — `floor(log2(|mid| / rad))`, the number of
//!   leading bits of the midpoint that are justified.
//! * [`ComplexBall::is_indeterminate`] — the radius is infinite.
//! * [`ComplexBall::value_if_accurate`] — a typed refusal unless the enclosure
//!   meets a stated bit requirement.

use rug::float::Round;
use rug::{Float, Rational};

use super::error::ThetaError;

/// A real number as `[mid ± rad]`, with the working precision that produced it.
///
/// Invariant: `rad >= 0`. `rad = +inf` marks an enclosure FLINT could not bound
/// — still a true statement about the value, just an empty one.
#[derive(Clone, Debug)]
pub struct RealBall {
    mid: Float,
    rad: Float,
    prec: u32,
}

impl RealBall {
    /// The exact ball `[v ± 0]`. An `f64` is a binary float, so nothing is lost.
    pub fn exact_f64(v: f64, prec: u32) -> Self {
        let p = prec.max(53);
        RealBall {
            mid: Float::with_val(p, v),
            rad: Float::with_val(p, 0.0),
            prec,
        }
    }

    /// The exact ball `[n ± 0]` for an integer that fits the precision.
    pub fn exact_i64(v: i64, prec: u32) -> Self {
        let p = prec.max(64);
        RealBall {
            mid: Float::with_val(p, v),
            rad: Float::with_val(p, 0.0),
            prec,
        }
    }

    /// `[mid ± rad]` built from two `rug::Float`s as given.
    ///
    /// The radius is taken as an absolute value; a negative one is a caller
    /// error that would otherwise produce an enclosure narrower than the truth.
    pub fn from_mid_rad(mid: Float, rad: Float, prec: u32) -> Self {
        let mut rad = rad;
        rad.abs_mut();
        RealBall { mid, rad, prec }
    }

    /// An enclosure of a rational, rounding the midpoint to `prec` bits and
    /// charging the rounding to the radius.
    pub fn from_rational(r: &Rational, prec: u32) -> Self {
        let p = prec.max(2);
        let (mid, ord) = Float::with_val_round(p, r, Round::Nearest);
        let rad = if ord == std::cmp::Ordering::Equal {
            Float::with_val(p, 0.0)
        } else {
            // One ulp of the midpoint: `get_exp` reports `e` with
            // `|mid| = m * 2^e`, `0.5 <= m < 1`, so `ulp = 2^(e - p)` and the
            // nearest-rounding error is at most half of that. A full ulp is
            // used, which is rigorous with a factor of two to spare.
            match mid.get_exp() {
                Some(e) => {
                    let mut u = Float::with_val(p, 1.0);
                    u <<= e - p as i32;
                    u
                }
                None => Float::with_val(p, 0.0),
            }
        };
        RealBall { mid, rad, prec }
    }

    /// `[0 ± inf]` — a true enclosure of anything, and an answer to nothing.
    pub fn indeterminate(prec: u32) -> Self {
        RealBall {
            mid: Float::with_val(prec.max(53), 0.0),
            rad: Float::with_val(prec.max(53), f64::INFINITY),
            prec,
        }
    }

    /// The midpoint. **Not** the value: see [`RealBall::accuracy_bits`].
    pub fn midpoint(&self) -> &Float {
        &self.mid
    }

    /// The radius, an upper bound on `|value - midpoint|`.
    pub fn radius(&self) -> &Float {
        &self.rad
    }

    /// The working precision, in bits, that produced this ball.
    pub fn precision(&self) -> u32 {
        self.prec
    }

    /// The midpoint as `f64`, rounded to nearest. Pair with
    /// [`RealBall::radius_f64`], which rounds outward, for a true `f64`
    /// enclosure.
    pub fn midpoint_f64(&self) -> f64 {
        self.mid.to_f64()
    }

    /// The radius as `f64`, rounded **up**, and inflated to absorb the error
    /// the midpoint takes on when it is rounded to `f64`.
    ///
    /// `[midpoint_f64() +/- radius_f64()]` is therefore a true enclosure on its
    /// own, which `[midpoint_f64(), radius.to_f64()]` would not be: rounding
    /// the midpoint to nearest moves it by up to half an `f64` ulp, and a
    /// radius that did not absorb that would be too small by exactly the amount
    /// nobody would notice.
    pub fn radius_f64(&self) -> f64 {
        if !self.rad.is_finite() {
            return f64::INFINITY;
        }
        let mid_d = self.mid.to_f64();
        if !mid_d.is_finite() {
            return f64::INFINITY;
        }
        // 64 guard bits past the midpoint's own width make the subtraction
        // exact: `mid_d` carries 53 significant bits at an exponent within one
        // of the midpoint's, so `mid - mid_d` needs at most `prec` bits.
        let p = self.mid.prec().max(64) + 64;
        let drift = (Float::with_val(p, &self.mid) - Float::with_val(p, mid_d)).abs();
        // Round the sum **up** rather than to nearest, so that the `f64` this
        // returns is an upper bound even in the case where the exact sum is not
        // representable at `p` bits.
        let (total, _) = Float::with_val_round(p, &self.rad + &drift, Round::Up);
        let up = total.to_f64_round(Round::Up);
        if up.is_finite() && up >= 0.0 {
            up
        } else {
            f64::INFINITY
        }
    }

    /// Is the radius infinite?
    pub fn is_indeterminate(&self) -> bool {
        !self.rad.is_finite() || !self.mid.is_finite()
    }

    /// Is the radius exactly zero?
    pub fn is_exact(&self) -> bool {
        self.rad.is_zero()
    }

    /// A **lower bound** on `floor(log2(|mid| / rad))` — how many leading bits
    /// of the midpoint the radius justifies.
    ///
    /// `i64::MAX` for an exact ball, `i64::MIN` when the radius is infinite or
    /// the midpoint is zero and the radius is not.
    ///
    /// FLINT's `arb_rel_accuracy_bits` is the raw exponent difference, which
    /// can overstate the true figure by one bit (both exponents are only known
    /// to within the half-open binade they name). One bit is subtracted here so
    /// that the number is never optimistic: this value gates
    /// [`ComplexBall::value_if_accurate`] and [`super::Precision::AccurateTo`],
    /// and an accuracy claim that is one bit too good is exactly the kind of
    /// wrong error bound that cannot be detected downstream. Expect results one
    /// lower than FLINT reports.
    pub fn accuracy_bits(&self) -> i64 {
        if self.is_indeterminate() {
            return i64::MIN;
        }
        if self.rad.is_zero() {
            return i64::MAX;
        }
        match (self.mid.get_exp(), self.rad.get_exp()) {
            (Some(me), Some(re)) => i64::from(me) - i64::from(re) - 1,
            _ => i64::MIN,
        }
    }

    /// The endpoints `(lo, hi)`, with `lo` rounded **towards** the interior and
    /// `hi` rounded towards the interior as well.
    ///
    /// Deliberately the *inner* rounding: every predicate below answers "is
    /// this certainly true", so each of them must be allowed to say `false`
    /// when it cannot tell, and never `true`. An outward-rounded endpoint would
    /// make `contains` and `overlaps` err in the direction of claiming more
    /// than is known. `interval_f64` is the outward-rounded counterpart, for
    /// reporting rather than for deciding.
    fn inner_endpoints(&self) -> Option<(Float, Float)> {
        if !self.mid.is_finite() || !self.rad.is_finite() {
            return None;
        }
        let p = self.mid.prec().max(self.rad.prec()).max(64) + 64;
        let (lo, _) = Float::with_val_round(p, &self.mid - &self.rad, Round::Up);
        let (hi, _) = Float::with_val_round(p, &self.mid + &self.rad, Round::Down);
        Some((lo, hi))
    }

    /// Does this ball **certainly** contain `v`?
    pub fn contains_f64(&self, v: f64) -> bool {
        if !self.rad.is_finite() {
            // `[anything +/- inf]` encloses every finite number, and the
            // non-finite midpoint case is the same statement.
            return true;
        }
        if !self.mid.is_finite() {
            return false;
        }
        match self.inner_endpoints() {
            Some((lo, hi)) => lo <= v && v <= hi,
            None => false,
        }
    }

    /// Do the two balls **certainly** have a point in common?
    ///
    /// The predicate to reach for when checking an identity between two
    /// computed results: it is true exactly when the identity is consistent
    /// with both error bounds. Like everything else here it errs towards
    /// `false`, so a `true` is a statement and a `false` is "not established".
    pub fn overlaps(&self, other: &RealBall) -> bool {
        if !self.rad.is_finite() || !other.rad.is_finite() {
            return true;
        }
        let (Some((a_lo, a_hi)), Some((b_lo, b_hi))) =
            (self.inner_endpoints(), other.inner_endpoints())
        else {
            return false;
        };
        a_lo <= b_hi && b_lo <= a_hi
    }

    /// Does this ball certainly contain zero?
    pub fn contains_zero(&self) -> bool {
        self.contains_f64(0.0)
    }

    /// `[mid - rad, mid + rad]` as a pair of `f64`s, rounded **outward** — the
    /// counterpart of `RealBall::inner_endpoints`, for reporting an enclosure
    /// rather than for deciding a predicate.
    pub fn interval_f64(&self) -> (f64, f64) {
        let m = self.midpoint_f64();
        let r = self.radius_f64();
        if !m.is_finite() || !r.is_finite() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let (lo, _) = Float::with_val_round(64, Float::with_val(64, m) - r, Round::Down);
        let (hi, _) = Float::with_val_round(64, Float::with_val(64, m) + r, Round::Up);
        (lo.to_f64_round(Round::Down), hi.to_f64_round(Round::Up))
    }

    /// The midpoint rendered with `digits` significant decimal digits.
    pub fn to_decimal_string(&self, digits: usize) -> String {
        if !self.mid.is_finite() {
            return format!("{}", self.mid);
        }
        format!("{:.*}", digits.max(1), self.mid)
    }
}

impl std::fmt::Display for RealBall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} +/- {:.3e}]",
            self.to_decimal_string(17),
            self.radius_f64()
        )
    }
}

/// A complex number as a pair of real balls: `(re ± r_re) + i (im ± r_im)`.
#[derive(Clone, Debug)]
pub struct ComplexBall {
    pub(crate) re: RealBall,
    pub(crate) im: RealBall,
}

impl ComplexBall {
    /// The exact ball `[re ± 0] + i [im ± 0]`.
    pub fn exact_f64(re: f64, im: f64, prec: u32) -> Self {
        ComplexBall {
            re: RealBall::exact_f64(re, prec),
            im: RealBall::exact_f64(im, prec),
        }
    }

    /// The exact ball at a Gaussian-integer point.
    pub fn exact_i64(re: i64, im: i64, prec: u32) -> Self {
        ComplexBall {
            re: RealBall::exact_i64(re, prec),
            im: RealBall::exact_i64(im, prec),
        }
    }

    /// An enclosure of a Gaussian rational.
    pub fn from_rationals(re: &Rational, im: &Rational, prec: u32) -> Self {
        ComplexBall {
            re: RealBall::from_rational(re, prec),
            im: RealBall::from_rational(im, prec),
        }
    }

    /// Assemble from two real balls.
    pub fn from_parts(re: RealBall, im: RealBall) -> Self {
        ComplexBall { re, im }
    }

    /// `[0 ± inf] + i [0 ± inf]`.
    pub fn indeterminate(prec: u32) -> Self {
        ComplexBall {
            re: RealBall::indeterminate(prec),
            im: RealBall::indeterminate(prec),
        }
    }

    /// The real part.
    pub fn real(&self) -> &RealBall {
        &self.re
    }

    /// The imaginary part.
    pub fn imag(&self) -> &RealBall {
        &self.im
    }

    /// Midpoint of the real part. **Not** the value — see
    /// [`ComplexBall::accuracy_bits`].
    pub fn midpoint_re(&self) -> &Float {
        self.re.midpoint()
    }

    /// Midpoint of the imaginary part.
    pub fn midpoint_im(&self) -> &Float {
        self.im.midpoint()
    }

    /// Radius of the real part.
    pub fn radius_re(&self) -> &Float {
        self.re.radius()
    }

    /// Radius of the imaginary part.
    pub fn radius_im(&self) -> &Float {
        self.im.radius()
    }

    /// The working precision, in bits, that produced this ball.
    pub fn precision(&self) -> u32 {
        self.re.precision()
    }

    /// Relative accuracy of the enclosure, in bits, measured against the larger
    /// component — the same convention as FLINT's `acb_rel_accuracy_bits`.
    pub fn accuracy_bits(&self) -> i64 {
        if self.is_indeterminate() {
            return i64::MIN;
        }
        if self.re.is_exact() && self.im.is_exact() {
            return i64::MAX;
        }
        let mid_exp = [self.re.midpoint().get_exp(), self.im.midpoint().get_exp()]
            .into_iter()
            .flatten()
            .max();
        let rad_exp = [self.re.radius().get_exp(), self.im.radius().get_exp()]
            .into_iter()
            .flatten()
            .max();
        match (mid_exp, rad_exp) {
            // One bit conservative, as in `RealBall::accuracy_bits`.
            (Some(me), Some(re)) => i64::from(me) - i64::from(re) - 1,
            (Some(_), None) => i64::MAX,
            _ => i64::MIN,
        }
    }

    /// Either component's radius is infinite.
    pub fn is_indeterminate(&self) -> bool {
        self.re.is_indeterminate() || self.im.is_indeterminate()
    }

    /// Both components are exact.
    pub fn is_exact(&self) -> bool {
        self.re.is_exact() && self.im.is_exact()
    }

    /// Does this ball certainly contain the point `re + i*im`?
    pub fn contains_f64(&self, re: f64, im: f64) -> bool {
        self.re.contains_f64(re) && self.im.contains_f64(im)
    }

    /// Does this ball certainly contain zero?
    pub fn contains_zero(&self) -> bool {
        self.re.contains_zero() && self.im.contains_zero()
    }

    /// Do the two balls **certainly** share a point? The test to use when
    /// checking an identity between two computed enclosures: it is true exactly
    /// when the identity is *consistent* with both error bounds, and it errs
    /// towards `false` rather than towards claiming agreement.
    pub fn overlaps(&self, other: &ComplexBall) -> bool {
        self.re.overlaps(&other.re) && self.im.overlaps(&other.im)
    }

    /// The midpoint as a pair of `f64`s, **but only if the enclosure is worth
    /// reading**.
    ///
    /// Refuses with [`ThetaError::InsufficientAccuracy`] when the relative
    /// accuracy is below `min_accuracy_bits`, and with
    /// [`ThetaError::Indeterminate`] when the radius is infinite. This is the
    /// accessor to reach for when the result is about to be compared, plotted
    /// or stored: it cannot hand back a midpoint with nothing behind it.
    pub fn value_if_accurate(
        &self,
        function: &'static str,
        min_accuracy_bits: u32,
    ) -> Result<(f64, f64), ThetaError> {
        if self.is_indeterminate() {
            return Err(ThetaError::Indeterminate { function });
        }
        let acc = self.accuracy_bits();
        if acc < i64::from(min_accuracy_bits) {
            return Err(ThetaError::InsufficientAccuracy {
                function,
                requested_bits: min_accuracy_bits,
                achieved_bits: Some(acc),
                at_precision: self.precision(),
            });
        }
        Ok((self.re.midpoint_f64(), self.im.midpoint_f64()))
    }
}

impl std::fmt::Display for ComplexBall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}*I", self.re, self.im)
    }
}
