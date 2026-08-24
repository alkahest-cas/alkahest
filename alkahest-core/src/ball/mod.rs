//! Phase 22 — Arbitrary-precision ball arithmetic with rigorous error bounds.
//!
//! Implements real ball arithmetic `[mid ± rad]` where `mid` and `rad` are
//! arbitrary-precision floating-point numbers (`rug::Float` / MPFR).  Every
//! operation guarantees that the true result lies within the output ball.
//!
//! # Relationship to FLINT 3.x / Arb
//!
//! FLINT 3.x merged the [Arb library](https://arblib.org/) which provides
//! `arb_t` (real balls) and `acb_t` (complex balls) in C.  This Rust module
//! implements the same mathematical contract:
//!
//! - `ArbBall ≈ arb_t` — a real number `mid ± rad` with MPFR precision.
//! - `AcbBall ≈ acb_t` — a complex number `(re ± r_re) + i(im ± r_im)`.
//!
//! When `libflint3-dev` becomes available as a system package, the
//! computation kernels in this module can be replaced with FFI calls to
//! `arb_add`, `arb_mul`, etc.  The public Rust API (`ArbBall`, `AcbBall`,
//! `IntervalEval`) will remain unchanged.
//!
//! # Rounding model
//!
//! All operations use **outward rounding**: the radius is grown by `2^{-prec}`
//! of the midpoint magnitude after each operation, ensuring the true result is
//! always contained in the ball.  The default precision is 128 bits.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::ball::{ArbBall, IntervalEval};
//! use alkahest_cas::kernel::{Domain, ExprPool};
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! let expr = pool.add(vec![
//!     pool.pow(x, pool.integer(2_i32)),  // x²
//!     pool.integer(1_i32),               // + 1
//! ]);
//!
//! // Evaluate x² + 1 at x ∈ [2.9, 3.1]  (ball centred at 3, radius 0.1)
//! let x_ball = ArbBall::from_midpoint_radius(3.0, 0.1, 128);
//! let mut eval = IntervalEval::new(128);
//! eval.bind(x, x_ball);
//! let result = eval.eval(expr, &pool).unwrap();
//! // True value: [2.9², 3.1²] + 1 = [9.41, 10.61]
//! // result.contains(9.5) should be true
//! assert!(result.contains(9.5));
//! assert!(result.contains(10.5));
//! ```

// Phase 29 — FLINT 3.x / Arb native bindings.
//
// Design: when `--features flint3` is enabled the arithmetic kernels below
// will be replaced with direct FFI to `arb_t` / `acb_t` (`arb_add`,
// `arb_mul`, `arb_sin`, …).  The public API (`ArbBall`, `AcbBall`,
// `IntervalEval`) is unchanged — only the backend swaps.
//
// Status: the MPFR-backed path is the unconditional implementation today.
// Ubuntu 24.04 ships `libflint3-dev` (FLINT ≥ 3.0); until that becomes the
// CI baseline the flint3 feature flag is a no-op that compiles without error.
// The upgrade path is: add `alkahest-core/src/flint/arb.rs` with `extern "C"`
// bindings, gate with `#[cfg(feature = "flint3")]`, verify all ball::tests
// pass, then confirm rad is tighter than the MPFR path on exp/sin tests.

use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::primitive::PrimitiveRegistry;
use rug::{float::Round, ops::Pow, Float};
use std::collections::HashMap;
use std::fmt;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Precision constant
// ---------------------------------------------------------------------------

/// Default precision in bits (matches Arb's default for `arb_t`).
pub const DEFAULT_PREC: u32 = 128;

// ---------------------------------------------------------------------------
// ArbBall — real ball [mid ± rad]
// ---------------------------------------------------------------------------

/// A real number represented as a ball `[mid - rad, mid + rad]`.
///
/// Invariants: `rad >= 0`.  If `rad = +inf` the ball represents an unknown
/// value (propagated from unsupported operations).
#[derive(Clone, Debug)]
pub struct ArbBall {
    pub mid: Float,
    pub rad: Float,
    pub prec: u32,
}

impl ArbBall {
    // ── constructors ─────────────────────────────────────────────────────

    pub fn new(prec: u32) -> Self {
        ArbBall {
            mid: Float::new(prec),
            rad: Float::new(prec),
            prec,
        }
    }

    pub fn from_f64(v: f64, prec: u32) -> Self {
        let mid = Float::with_val(prec, v);
        // Conversion error ≤ 2^(exponent - prec)
        let rad = Float::with_val(prec, 0.0);
        ArbBall { mid, rad, prec }
    }

    pub fn from_midpoint_radius(mid: f64, rad: f64, prec: u32) -> Self {
        ArbBall {
            mid: Float::with_val(prec, mid),
            rad: Float::with_val(prec, rad.abs()),
            prec,
        }
    }

    pub fn from_integer(n: &rug::Integer, prec: u32) -> Self {
        ArbBall {
            mid: Float::with_val(prec, n),
            rad: Float::with_val(prec, 0.0),
            prec,
        }
    }

    pub fn from_rational(r: &rug::Rational, prec: u32) -> Self {
        // mid = round(r),  rad = |r - mid| ≤ 2^(exp-prec)
        let mid = Float::with_val(prec, r);
        let exact = Float::with_val(prec * 2, r);
        let diff = Float::with_val(prec, &exact - &mid).abs();
        ArbBall {
            mid,
            rad: diff,
            prec,
        }
    }

    pub fn infinity(prec: u32) -> Self {
        let inf = Float::with_val(prec, f64::INFINITY);
        ArbBall {
            mid: Float::new(prec),
            rad: inf,
            prec,
        }
    }

    // ── predicates ───────────────────────────────────────────────────────

    /// True if the ball is a single point (radius = 0).
    pub fn is_exact(&self) -> bool {
        self.rad == 0
    }

    /// True if `v` is contained in `[mid - rad, mid + rad]`.
    pub fn contains(&self, v: f64) -> bool {
        let v = Float::with_val(self.prec, v);
        let lo = Float::with_val(self.prec, &self.mid - &self.rad);
        let hi = Float::with_val(self.prec, &self.mid + &self.rad);
        v >= lo && v <= hi
    }

    /// Lower bound of the interval, rounded **down**.
    ///
    /// `mid - rad` does not generally fit in `prec` bits, and rounding it to
    /// nearest would hand back a bound *above* the interval's true lower end —
    /// which every kernel that starts from `lo()` then evaluates at, giving a
    /// hull that is an ulp too narrow at the bottom.  The same, mirrored, for
    /// [`ArbBall::hi`].
    pub fn lo(&self) -> Float {
        Float::with_val_round(self.prec, &self.mid - &self.rad, Round::Down).0
    }

    /// Upper bound of the interval, rounded **up**.  See [`ArbBall::lo`].
    pub fn hi(&self) -> Float {
        Float::with_val_round(self.prec, &self.mid + &self.rad, Round::Up).0
    }

    /// Midpoint as f64 (lossy).
    pub fn mid_f64(&self) -> f64 {
        self.mid.to_f64()
    }

    /// Radius as f64 (lossy).
    pub fn rad_f64(&self) -> f64 {
        self.rad.to_f64()
    }

    // ── arithmetic ───────────────────────────────────────────────────────

    /// Grow the radius to absorb the rounding of *both* fields: by
    /// `(|mid| + rad) · 2^{-(prec-3)}`.
    ///
    /// This used to bump by `|mid| · 2^{-prec}`, which covers the rounding of
    /// `mid` and nothing else.  A kernel that also rounds its `rad` — every
    /// endpoint hull, `cosh`, `erf`, `Mul` — can then report a ball an ulp
    /// narrower than the truth, and on a ball whose radius dwarfs its midpoint
    /// (`cosh([5.2, 9.2])`, `log([1/2, 2])`) the shortfall is the whole error.
    /// The eight-ulp headroom covers the handful of roundings the widest of
    /// those kernels performs (`Mul` rounds five times) and is still 10³⁴
    /// below `f64` resolution at the default 128 bits.
    fn add_rounding_error(&mut self) {
        if self.mid.is_infinite() || self.mid.is_nan() {
            self.rad = Float::with_val(self.prec, f64::INFINITY);
            return;
        }
        let mut scale = Float::with_val(
            self.prec,
            Float::with_val(self.prec, self.mid.abs_ref()) + &self.rad,
        );
        scale >>= self.prec.saturating_sub(3);
        self.rad += &scale;
    }
}

impl fmt::Display for ArbBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6} ± {:.2e}]", self.mid.to_f64(), self.rad.to_f64())
    }
}

impl PartialEq for ArbBall {
    /// Two balls are equal if their midpoints and radii are equal.
    fn eq(&self, other: &Self) -> bool {
        self.mid == other.mid && self.rad == other.rad
    }
}

// ── Arithmetic traits ────────────────────────────────────────────────────────

impl std::ops::Add for ArbBall {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let prec = self.prec.max(rhs.prec);
        let mid = Float::with_val(prec, &self.mid + &rhs.mid);
        let rad = Float::with_val(prec, &self.rad + &rhs.rad);
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }
}

impl std::ops::Sub for ArbBall {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let prec = self.prec.max(rhs.prec);
        let mid = Float::with_val(prec, &self.mid - &rhs.mid);
        let rad = Float::with_val(prec, &self.rad + &rhs.rad);
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }
}

impl std::ops::Mul for ArbBall {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let prec = self.prec.max(rhs.prec);
        // |a*b| ≤ |a|*|b|
        // rad(a*b) = |mid_a|*rad_b + |mid_b|*rad_a + rad_a*rad_b
        let mid = Float::with_val(prec, &self.mid * &rhs.mid);
        let ma = Float::with_val(prec, self.mid.abs_ref());
        let mb = Float::with_val(prec, rhs.mid.abs_ref());
        let rad = Float::with_val(prec, &ma * &rhs.rad)
            + Float::with_val(prec, &mb * &self.rad)
            + Float::with_val(prec, &self.rad * &rhs.rad);
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }
}

impl std::ops::Neg for ArbBall {
    type Output = Self;
    fn neg(self) -> Self {
        ArbBall {
            mid: -self.mid,
            rad: self.rad,
            prec: self.prec,
        }
    }
}

impl std::ops::Div for ArbBall {
    type Output = Option<Self>;
    fn div(self, rhs: Self) -> Option<Self> {
        if rhs.contains(0.0) {
            return None; // Division by zero / interval containing zero
        }
        let prec = self.prec.max(rhs.prec);
        // Monotone on positive/negative intervals
        let lo_rhs = rhs.lo();
        let hi_rhs = rhs.hi();
        // Compute all 4 corners, at working precision so the reduction to
        // `prec` happens once, outward, in `from_endpoints` below.
        // (`saturating_add`, not `+`: `clippy::suspicious_arithmetic_impl`
        // flags a bare `+` anywhere inside a `Div` impl.)
        let work = prec.saturating_add(32);
        let quot = |a: &Float, b: &Float| Float::with_val(work, a) / Float::with_val(work, b);
        let corners = [
            quot(&self.lo(), &lo_rhs),
            quot(&self.lo(), &hi_rhs),
            quot(&self.hi(), &lo_rhs),
            quot(&self.hi(), &hi_rhs),
        ];
        // `∞/∞` is NaN, so an unbounded operand makes the corner ordering
        // partial and `partial_cmp(...).unwrap()` panics. `None` is the
        // interface's existing "no enclosure" answer; a panic here crosses the
        // FFI boundary as a `BaseException` that `except Exception` misses.
        if corners.iter().any(|c| c.is_nan()) {
            return None;
        }
        let min = corners
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .clone();
        let max = corners
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .clone();
        // Outward rounding: the corners were computed at `work` bits and are
        // being reported at `prec`.
        Some(ArbBall::from_endpoints(&min, &max, prec))
    }
}

impl ArbBall {
    /// Integer power: `self^n` (n ≥ 0).
    pub fn powi(&self, n: i64) -> Self {
        if n == 0 {
            return ArbBall::from_f64(1.0, self.prec);
        }
        if n < 0 {
            // 1 / self^|n|
            let pos = self.powi(-n);
            return (ArbBall::from_f64(1.0, self.prec) / pos)
                .unwrap_or_else(|| ArbBall::infinity(self.prec));
        }
        // Fast exponentiation by squaring
        let mut result = ArbBall::from_f64(1.0, self.prec);
        let mut base = self.clone();
        let mut exp = n as u64;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exp >>= 1;
        }
        result
    }

    /// `self ^ exp` for a real exponent *ball*.
    ///
    /// # When the four-corner hull is legitimate, and when it is not
    ///
    /// This used to hull `base^exp` over the four corners of
    /// `[lo_b, hi_b] × [lo_e, hi_e]` for every input a single guard let
    /// through, `lo < 0 && !(exp.is_exact() && exp.lo().is_integer())`.  A
    /// corner hull encloses the range only where `x ↦ x^y` is monotone in `x`,
    /// and that guard admits the two cases where it is not:
    ///
    /// * an **even** integer exponent on a base straddling `0` — `x²` over
    ///   `[-1, 3]` was reported as `[1, 9]`, missing `x = 0 ↦ 0`;
    /// * a **negative** integer exponent on such a base — `x⁻²` over `[-1, 3]`
    ///   was reported as the *finite* `[0.111…, 1]` for a function that is
    ///   unbounded on the box (`0.001⁻² = 10⁶`).
    ///
    /// (`x ** 2` written with an `Integer` exponent node never reached here —
    /// `IntervalEval::eval_node` routes it to [`ArbBall::powi`] — but the
    /// same exponent arriving as a `Float`, a `Rational`, or a bound symbol
    /// did.)
    ///
    /// So the corner hull is kept only on a base that stays `≥ 0`, where
    /// `x^y = exp(y·ln x)` is monotone in `x` for fixed `y` *and* monotone in
    /// `y` for fixed `x`; a function monotone in each variable separately
    /// attains its extrema over a box at a corner.  Everything else with a
    /// real value — a base that reaches `0` or below, which forces an
    /// integer-valued exponent — goes to [`ArbBall::powi`], which is repeated
    /// ball multiplication and assumes no monotonicity at all (and whose
    /// reciprocal step returns `∞` exactly when the denominator straddles 0).
    /// What is left is complex-valued, and widens to `∞`.
    pub fn pow_f(&self, exp: &ArbBall) -> Self {
        let prec = self.prec;
        let lo = self.lo();
        let hi = self.hi();
        let e_lo = exp.lo();
        let e_hi = exp.hi();
        if !(lo.is_finite() && hi.is_finite() && e_lo.is_finite() && e_hi.is_finite()) {
            return ArbBall::infinity(prec);
        }
        // `lo == 0` joins the monotonicity argument as long as the exponent
        // stays ≥ 0: `0^y` is 0 for `y > 0` and 1 for `y = 0`, still monotone
        // in `y`.  With a negative exponent in reach it is a pole instead.
        if lo > 0 || (lo == 0 && e_lo >= 0) {
            let work = prec + 32;
            let corner =
                |b: &Float, e: &Float| Float::with_val(work, b).pow(Float::with_val(work, e));
            let corners = [
                corner(&lo, &e_lo),
                corner(&lo, &e_hi),
                corner(&hi, &e_lo),
                corner(&hi, &e_hi),
            ];
            // Defence in depth: a NaN corner makes the ordering partial and
            // `partial_cmp(...).unwrap()` would panic on it — a Rust panic
            // crossing the FFI boundary is a `BaseException` that an
            // `except Exception` handler does not catch.  An infinite corner
            // (overflow) is a true statement about an unbounded box, and `∞`
            // is its enclosure.
            if corners.iter().any(|c| !c.is_finite()) {
                return ArbBall::infinity(prec);
            }
            let min = corners
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .clone();
            let max = corners
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .clone();
            // Outward rounding: the corners were computed at `work` bits and
            // are being reported at `prec`.
            return ArbBall::from_endpoints(&min, &max, prec);
        }
        // The base reaches 0 or below.  A real value then requires an exponent
        // that is an exact integer; `x^(3/2)` arrives here as an exact point
        // ball at 1.5 and `(-3.3)^1.5` is not real, so `is_exact` alone is not
        // the test.
        if exp.is_exact() && e_lo.is_integer() {
            if let Some(n) = e_lo.to_integer().and_then(|n| n.to_i64()) {
                return self.powi(n);
            }
        }
        ArbBall::infinity(prec) // complex result possible, or unbounded
    }

    pub fn sin(&self) -> Self {
        // |sin(x)| ≤ 1, Lipschitz constant = 1
        // sin([m-r, m+r]) ⊆ [sin(m) - r, sin(m) + r]
        let prec = self.prec;
        let mid = Float::with_val(prec, self.mid.clone().sin());
        let rad = self.rad.clone();
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    pub fn cos(&self) -> Self {
        let prec = self.prec;
        let mid = Float::with_val(prec, self.mid.clone().cos());
        let rad = self.rad.clone();
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    pub fn exp(&self) -> Self {
        // e^[m-r, m+r] = [e^(m-r), e^(m+r)]
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).exp();
        let hi = Float::with_val(work, self.hi()).exp();
        // `from_endpoints` rounds outward.  Without that the ball is
        // exact-looking (`rad == 0`) for an exact input, which is a false
        // rigorous claim about a transcendental value; and on a ball whose
        // radius dwarfs its midpoint the endpoints' own rounding is not
        // covered by the `|mid|·2^-prec` bump `add_rounding_error` applies.
        ArbBall::from_endpoints(&lo, &hi, prec)
    }

    pub fn log(&self) -> Option<Self> {
        if self.lo() <= 0 {
            return None; // log undefined for non-positive values
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).ln();
        let hi = Float::with_val(work, self.hi()).ln();
        // Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&lo, &hi, prec))
    }

    pub fn sqrt(&self) -> Option<Self> {
        if self.lo() < 0 {
            return None;
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).sqrt();
        let hi = Float::with_val(work, self.hi()).sqrt();
        // Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&lo, &hi, prec))
    }

    /// tan([m-r, m+r]).  `None` only for a non-finite argument; a box that may
    /// contain a pole widens to `[-∞, ∞]`, which is what `tan` does there.
    ///
    /// # Why the pole test is an enclosure of `cos`, not a test on the endpoints
    ///
    /// This used to guard the pole with `near_pole(lo) || near_pole(hi)` — an
    /// `f64` test of `v % π` at the two endpoints — plus "`tan(lo) > tan(hi)`
    /// means we crossed a pole".  Neither sees a box that crosses a pole and
    /// comes back out the other side in increasing order: on
    /// `[0.1, 3.3415926535897933]` the endpoint hull is `[0.1003…, 0.2027…]`
    /// while `tan(1.5) = 14.101…` and `1.5` is in the box.  (`v % π` is not a
    /// pole test at all once `|v|` is large enough that `v % π` loses the low
    /// bits that decide the question.)
    ///
    /// `tan′ = sec² > 0`, so an endpoint hull *is* the exact range — but only
    /// on an interval where `cos` does not vanish, because those intervals are
    /// exactly the ones on which `tan` is continuous and increasing.  That is
    /// the rule the Taylor path already applies one module over
    /// (`validated::taylor::TaylorModel::tan` refuses when the `cos` enclosure
    /// contains zero, `E-VALIDATED-003`), and this is the same rule in ball
    /// form: [`ArbBall::cos`] encloses the range of `cos` over the box, so
    /// `0 ∉ cos([lo, hi])` *proves* the box lies inside a single branch.
    pub fn tan(&self) -> Option<Self> {
        let prec = self.prec;
        let lo = self.lo();
        let hi = self.hi();
        if !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        // Pole test.  A zero in the `cos` enclosure is the only way `tan` can
        // fail to be continuous and increasing on `[lo, hi]`; it is also the
        // only way it can be unbounded there, so `[-∞, ∞]` is the enclosure.
        if self.cos().contains(0.0) {
            return Some(ArbBall::infinity(prec));
        }
        let work = prec + 32;
        let lo_tan = Float::with_val(work, &lo).tan();
        let hi_tan = Float::with_val(work, &hi).tan();
        // `from_endpoints` rounds outward, which both absorbs the reduction
        // from `work` to `prec` and keeps an exact input from reporting
        // `rad == 0` — a false claim that a transcendental value is exactly
        // representable.
        Some(ArbBall::from_endpoints(&lo_tan, &hi_tan, prec))
    }

    pub fn sinh(&self) -> Self {
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).sinh();
        let hi = Float::with_val(work, self.hi()).sinh();
        // Rounded outward; see `exp`.
        ArbBall::from_endpoints(&lo, &hi, prec)
    }

    pub fn cosh(&self) -> Self {
        let prec = self.prec;
        // cosh is even and has a minimum at 0; handle by evaluating at lo, hi, and 0 if in range
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).cosh();
        let hi = Float::with_val(work, self.hi()).cosh();
        let (min_val, max_val) = if self.lo() <= 0 && self.hi() >= 0 {
            // minimum is cosh(0) = 1
            let cosh_lo = lo.clone();
            let cosh_hi = hi.clone();
            let min = Float::with_val(work, 1_f64);
            let max = if cosh_lo > cosh_hi { cosh_lo } else { cosh_hi };
            (min, max)
        } else if lo < hi {
            (lo, hi)
        } else {
            (hi, lo)
        };
        // Rounded outward; see `exp`.
        ArbBall::from_endpoints(&min_val, &max_val, prec)
    }

    pub fn tanh(&self) -> Self {
        // tanh is monotone, maps ℝ → (-1, 1)
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).tanh();
        let hi = Float::with_val(work, self.hi()).tanh();
        // Rounded outward; see `exp`.
        ArbBall::from_endpoints(&lo, &hi, prec)
    }

    pub fn asin(&self) -> Option<Self> {
        if self.lo() < -1 || self.hi() > 1 {
            return None;
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).asin();
        let hi = Float::with_val(work, self.hi()).asin();
        // Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&lo, &hi, prec))
    }

    pub fn acos(&self) -> Option<Self> {
        if self.lo() < -1 || self.hi() > 1 {
            return None;
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).acos();
        let hi = Float::with_val(work, self.hi()).acos();
        // acos is decreasing, so lo/hi swap; `from_endpoints` takes either
        // order.  Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&hi, &lo, prec))
    }

    pub fn atan(&self) -> Self {
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).atan();
        let hi = Float::with_val(work, self.hi()).atan();
        // Rounded outward; see `exp`.
        ArbBall::from_endpoints(&lo, &hi, prec)
    }

    /// asinh([m-r, m+r]) — monotone increasing on all of ℝ.
    pub fn asinh(&self) -> Self {
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).asinh();
        let hi = Float::with_val(work, self.hi()).asinh();
        // Rounded outward; see `exp`.
        ArbBall::from_endpoints(&lo, &hi, prec)
    }

    /// acosh([m-r, m+r]) — monotone increasing on `[1, ∞)`. Returns `None` if
    /// the interval extends below 1 (outside the real domain).
    pub fn acosh(&self) -> Option<Self> {
        if self.lo() < 1 {
            return None;
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).acosh();
        let hi = Float::with_val(work, self.hi()).acosh();
        // Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&lo, &hi, prec))
    }

    /// atanh([m-r, m+r]) — monotone increasing on `(-1, 1)`. Returns `None` if
    /// the interval reaches or leaves `(-1, 1)` (outside the real domain).
    pub fn atanh(&self) -> Option<Self> {
        if self.lo() <= -1 || self.hi() >= 1 {
            return None;
        }
        let prec = self.prec;
        let work = prec + 32;
        let lo = Float::with_val(work, self.lo()).atanh();
        let hi = Float::with_val(work, self.hi()).atanh();
        // Rounded outward; see `exp`.
        Some(ArbBall::from_endpoints(&lo, &hi, prec))
    }

    pub fn erf(&self) -> Self {
        let prec = self.prec;
        // Use midpoint + Lipschitz: |erf'(x)| = 2/sqrt(π) * exp(-x²) ≤ 2/sqrt(π) ≈ 1.13
        let mid = Float::with_val(prec, self.mid.clone().erf());
        let rad = Float::with_val(prec, &self.rad * &erf_lipschitz(prec));
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    pub fn erfc(&self) -> Self {
        let prec = self.prec;
        let mid = Float::with_val(prec, self.mid.clone().erfc());
        let rad = Float::with_val(prec, &self.rad * &erf_lipschitz(prec));
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    /// Re-round an arbitrary `Float` into a ball at this precision, outward.
    ///
    /// `add_rounding_error` alone is only valid for a `mid` that was *computed*
    /// at `prec`; a value carried in at higher precision has to have the
    /// truncation itself absorbed first.
    fn from_point(v: &Float, prec: u32) -> Self {
        let mid = Float::with_val(prec, v);
        let rad = Float::with_val(prec, Float::with_val(prec + 32, v - &mid).abs());
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    /// Enclosure of `[lo, hi]`, rounding outward.  `lo <= hi` is the caller's
    /// business; a swapped pair yields the same (still enclosing) ball.
    fn from_endpoints(lo: &Float, hi: &Float, prec: u32) -> Self {
        let mid = Float::with_val(prec, Float::with_val(prec + 32, lo + hi) / 2u32);
        let mut rad = Float::with_val(prec, Float::with_val(prec + 32, hi - lo) / 2u32).abs();
        // `add_rounding_error` alone bumps by `|mid|·2⁻ᵖʳᵉᶜ`, which does not
        // cover the rounding of `rad` itself on a ball whose radius dwarfs its
        // midpoint (`[-1, 1+ε]` has `|mid| ≈ ε/2` and `rad ≈ 1`). Bump by the
        // ball's whole magnitude instead, so both roundings are absorbed.
        let mut bump = Float::with_val(prec, Float::with_val(prec, mid.abs_ref()) + &rad);
        bump >>= prec.saturating_sub(2);
        rad += bump;
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    /// Principal-branch Lambert W₀.  Domain: `x ≥ −1/e`.
    ///
    /// # Why this does not simply hull two `f64` evaluations
    ///
    /// It used to.  `crate::special::lambert_w0` is an `f64` Halley iteration,
    /// so its answer carries ~10⁻¹⁶ of error, while the ball this built claimed
    /// a radius of `|mid|·2⁻ᵖʳᵉᶜ` — 5·10⁻⁴⁰ at the default 128 bits.  On the
    /// degenerate ball `[1, 1]` that is an enclosure of width 10⁻³⁹ centred
    /// 3·10⁻¹⁷ away from `W₀(1) = 0.567143290409783873…`: an interval that does
    /// not contain the value it encloses, which is the one failure mode a
    /// certificate subsystem cannot have.
    ///
    /// # What replaces it
    ///
    /// `W₀` is the inverse of `g(w) = w·eʷ` on `w ≥ −1`, where `g` is strictly
    /// increasing (`g′(w) = (1+w)eʷ > 0` for `w > −1`).  Monotonicity turns a
    /// bound on `W₀` into a *checkable* statement about `g`:
    ///
    /// ```text
    /// for v, u ≥ −1:   g(v) ≤ x  ⟹  W₀(x) ≥ v,      g(u) ≥ x  ⟹  W₀(x) ≤ u.
    /// ```
    ///
    /// So the iteration below is only ever a *guess*: the returned bracket is
    /// certified afterwards by evaluating `g` in ball arithmetic at the two
    /// candidate endpoints and checking those two inequalities outward.  A
    /// wrong or badly converged guess can only widen the answer, never
    /// invalidate it.  `W₀ ≥ −1` holds on the whole principal branch by
    /// definition, so `−1` is always available as a fallback lower bound.
    ///
    /// The enclosure over a ball is then `[low(lo), high(hi)]` — an endpoint
    /// hull, which is valid **here** precisely because `W₀` is monotone on its
    /// domain (contrast [`ArbBall::bessel_jn`], where it was not).
    pub fn lambert_w0(&self) -> Option<Self> {
        let prec = self.prec;
        let (low, _) = lambert_w0_bracket(&self.lo(), prec)?;
        let (_, high) = lambert_w0_bracket(&self.hi(), prec)?;
        Some(ArbBall::from_endpoints(&low, &high, prec))
    }

    /// `Γ(x)` for a ball lying strictly inside `(0, ∞)`.  `None` otherwise —
    /// `Γ` has poles at every non-positive integer, and the reflection formula
    /// that would cover the gaps between them is not implemented here.
    ///
    /// # Enclosure
    ///
    /// `Γ″(x) = ∫₀^∞ t^{x−1}(ln t)² e^{−t} dt > 0` on `(0, ∞)`, so `Γ` is
    /// **convex** there.  That gives both ends of the enclosure without any
    /// monotonicity assumption:
    ///
    /// * a convex function on `[a, b]` attains its maximum at an endpoint, so
    ///   `max Γ = max(Γ(a), Γ(b))`;
    /// * a convex function lies above each of its tangents, so with
    ///   `Γ′ = ψ·Γ` both `T_a(x) = Γ(a) + Γ(a)ψ(a)(x−a)` and
    ///   `T_b(x) = Γ(b) + Γ(b)ψ(b)(x−b)` are lower bounds on all of `[a, b]`,
    ///   and each is minimised over `[a, b]` at one of the two endpoints.
    ///
    /// The larger of the two tangent minima is kept, floored at `0` because
    /// `Γ > 0` on `(0, ∞)`.  Both bounds are exact in the limit `b → a`, so a
    /// subdivided box converges; neither assumes `Γ` is monotone, which it is
    /// not (its minimum sits at `x ≈ 1.4616`, inside the range that matters).
    pub fn gamma(&self) -> Option<Self> {
        let prec = self.prec;
        let a = self.lo();
        let b = self.hi();
        if !(a.is_finite() && b.is_finite()) || a <= 0 {
            return None;
        }
        let work = prec + 32;
        let ga = ArbBall::from_point(&Float::with_val(work, &a).gamma(), prec);
        let gb = ArbBall::from_point(&Float::with_val(work, &b).gamma(), prec);
        let mut psi_a = Float::with_val(work, &a);
        psi_a.digamma_mut();
        let mut psi_b = Float::with_val(work, &b);
        psi_b.digamma_mut();
        let width = ArbBall::from_point(&Float::with_val(work, &b - &a), prec);

        // max: convexity puts it at an endpoint.
        let upper = {
            let (x, y) = (ga.hi(), gb.hi());
            if x > y {
                x
            } else {
                y
            }
        };
        // min: each tangent line is a lower bound for Γ on all of [a, b], and
        // a line is minimised over an interval at one of its endpoints —
        // `T_a(a) = Γ(a)` and `T_a(b)`, `T_b(b) = Γ(b)` and `T_b(a)`.  The
        // *largest* of the two per-line minima is the sharpest bound that
        // follows.  `min(Γ(a), Γ(b))` on its own would not be a lower bound at
        // all: Γ dips below both endpoints whenever the box straddles 1.4616.
        let ta_far = ga.clone() + ga.clone() * ArbBall::from_point(&psi_a, prec) * width.clone();
        let tb_far = gb.clone() - gb.clone() * ArbBall::from_point(&psi_b, prec) * width;
        let line_min = |near: Float, far: Float| if near < far { near } else { far };
        let from_a = line_min(ga.lo(), ta_far.lo());
        let from_b = line_min(gb.lo(), tb_far.lo());
        let mut lower = if from_a > from_b { from_a } else { from_b };
        // Γ > 0 on (0, ∞), so a tangent bound that has gone negative on a wide
        // box is superseded by 0.
        if lower < 0 {
            lower = Float::new(prec);
        }
        Some(ArbBall::from_endpoints(&lower, &upper, prec))
    }

    /// Digamma ψ(x).  Returns `None` when the ball contains a non-positive
    /// integer pole.
    pub fn digamma(&self) -> Option<Self> {
        let prec = self.prec;
        let lo = self.lo();
        let hi = self.hi();
        if !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        // ψ is increasing between consecutive poles (ψ′ = trigamma > 0), so the
        // endpoint hull is the range as long as no pole is enclosed.  The poles
        // sit at the non-positive integers, so it is enough to look at the
        // smallest integer in `[lo, hi]` — iterating over every integer in the
        // box, as this used to, walks 10¹⁸ steps on a box like `[1, 1e18]`.
        let k_min = Float::with_val(prec, lo.ceil_ref());
        if k_min <= hi && k_min <= 0 {
            return None;
        }
        let work = prec + 32;
        let mut flo = Float::with_val(work, &lo);
        flo.digamma_mut();
        let mut fhi = Float::with_val(work, &hi);
        fhi.digamma_mut();
        // Rounded outward; see `exp`.  (The endpoints also used to be reduced
        // to `f64` before evaluation, which moves them *inside* the box and can
        // shrink the hull below the true range.)
        Some(ArbBall::from_endpoints(&flo, &fhi, prec))
    }

    /// Bessel function of the first kind Jₙ(x) for integer order `n`.
    ///
    /// # Why this is midpoint + Lipschitz rather than an endpoint hull
    ///
    /// Jₙ oscillates, so `hull(Jₙ(lo), Jₙ(hi))` is **not** an enclosure of its
    /// range: on `[-1, 1]` the two endpoints agree (`J₀(±1) ≈ 0.7652`) and the
    /// hull collapses to a point that excludes `J₀(0) = 1`. An endpoint hull is
    /// only valid for a monotone function, which every other kernel here that
    /// uses one (`exp`, `log`, `sqrt`, `tanh`, the inverse trig family,
    /// `digamma` between its poles, `floor`, `ceil`) is on its stated domain.
    ///
    /// The mean value theorem gives a sound enclosure instead:
    /// `|Jₙ(x) − Jₙ(m)| ≤ L·|x − m|` with `L = sup|Jₙ′|`. For every integer `n`
    /// and every real `x`, `|Jₙ(x)| ≤ 1`; with `J₀′ = −J₁` and
    /// `Jₙ′ = (Jₙ₋₁ − Jₙ₊₁)/2` for `n ≥ 1` that yields `|Jₙ′| ≤ 1`, so `L = 1`
    /// is rigorous at every order (and `J₋ₙ = (−1)ⁿ Jₙ` covers negative `n`).
    /// The bound is loose — the true suprema are ≈ 0.582 for `J₀` and ≈ 0.582
    /// for `J₁` — but it is a *bound*, which the hull was not.
    pub fn bessel_jn(&self, n: i32) -> Self {
        let prec = self.prec;
        // Evaluate at the midpoint at full working precision (MPFR's `jn` is
        // correctly rounded, so the error is under half an ulp and is absorbed
        // by `add_rounding_error` below).
        let mut mid = Float::with_val(prec, &self.mid);
        mid.jn_mut(n);
        let mut b = ArbBall {
            mid,
            rad: self.rad.clone(),
            prec,
        };
        b.add_rounding_error();
        b
    }

    pub fn abs_ball(&self) -> Self {
        let prec = self.prec;
        // |[m-r, m+r]| — if interval straddles zero the lower bound is 0
        if self.lo() <= 0 && self.hi() >= 0 {
            let max_abs = self.lo().abs().max(&self.hi().abs()).clone();
            ArbBall {
                mid: max_abs.clone() / 2_f64,
                rad: max_abs / 2_f64,
                prec,
            }
        } else {
            let mid = Float::with_val(prec, self.mid.clone().abs());
            let rad = self.rad.clone();
            let mut b = ArbBall { mid, rad, prec };
            b.add_rounding_error();
            b
        }
    }

    pub fn floor_ball(&self) -> Self {
        let prec = self.prec;
        let lo_floor = Float::with_val(prec, self.lo().floor());
        let hi_floor = Float::with_val(prec, self.hi().floor());
        let diff = Float::with_val(prec, &hi_floor - &lo_floor);
        let sum = Float::with_val(prec, &lo_floor + &hi_floor);
        let mut b = ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        };
        // Endpoints are rounded to `prec`; see `exp`.
        b.add_rounding_error();
        b
    }

    pub fn ceil_ball(&self) -> Self {
        let prec = self.prec;
        let lo_ceil = Float::with_val(prec, self.lo().ceil());
        let hi_ceil = Float::with_val(prec, self.hi().ceil());
        let diff = Float::with_val(prec, &hi_ceil - &lo_ceil);
        let sum = Float::with_val(prec, &lo_ceil + &hi_ceil);
        let mut b = ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        };
        // Endpoints are rounded to `prec`; see `exp`.
        b.add_rounding_error();
        b
    }
}

/// `2/√π ≈ 1.1283791…`, the Lipschitz constant of `erf` and `erfc`, rounded
/// **up**.  `Float::with_val(prec, 2.0/π.sqrt())` rounds to nearest, and a
/// Lipschitz constant rounded down is not a Lipschitz constant.
fn erf_lipschitz(prec: u32) -> Float {
    let work = prec + 32;
    let two_over_sqrt_pi = Float::with_val(
        work,
        2u32 / Float::with_val(work, rug::float::Constant::Pi).sqrt(),
    );
    let mut v = Float::with_val(prec, &two_over_sqrt_pi);
    // Round up: nudge by an ulp unless the reduction was exact.
    if v < two_over_sqrt_pi {
        v.next_up();
    }
    v
}

/// A certified bracket `(low, high)` with `low ≤ W₀(x) ≤ high`, or `None` when
/// `x` is outside the principal branch's domain `x ≥ −1/e`.
///
/// The certificate is the monotonicity of `g(w) = w·eʷ` on `w ≥ −1`, spelled
/// out on [`ArbBall::lambert_w0`]: a candidate `v ≥ −1` is admitted as a lower
/// bound exactly when `g(v) ≤ x` is *proved* by an outward ball evaluation of
/// `g`, and likewise `u` as an upper bound when `g(u) ≥ x`.  Nothing about how
/// the candidates were produced enters the argument, so the Newton iteration
/// that produces them needs no error analysis of its own.
///
/// There is no `−1/e` constant anywhere here, and deliberately so: for
/// `x < −1/e` no `u ≥ −1` satisfies `g(u) ≥ x`… — rather, *every* `u` does
/// (the minimum of `g` is `−1/e > x`), but then no `v` satisfies `g(v) ≤ x`,
/// and the lower search falls through to `−1`, whose own check `g(−1) ≤ x`
/// fails.  The domain test is therefore the bracket search itself, which
/// cannot disagree with the arithmetic the way a rounded literal can.
fn lambert_w0_bracket(x: &Float, prec: u32) -> Option<(Float, Float)> {
    if !x.is_finite() {
        return None;
    }
    let work = prec + 64;
    let minus_one = Float::with_val(work, -1);

    // Guess.  `f64` when the argument fits, the large-`x` asymptote otherwise;
    // either way this is only a starting point for the certified search below.
    let xf = x.to_f64();
    let mut w = match (xf.is_finite(), crate::special::lambert_w0(xf)) {
        (true, Some(v)) if v.is_finite() => Float::with_val(work, v),
        _ if *x > 1 => {
            let l = Float::with_val(work, x).ln();
            let ll = Float::with_val(work, l.clone().ln());
            l - ll
        }
        _ => Float::with_val(work, 0),
    };
    // Newton on g(w) − x.  Quadratic away from the branch point, and merely
    // slow (never wrong) at it, because the outcome is checked afterwards.
    for _ in 0..48 {
        let ew = Float::with_val(work, w.clone().exp());
        let num = Float::with_val(work, Float::with_val(work, &w * &ew) - x);
        let den = Float::with_val(work, &ew * Float::with_val(work, &w + 1u32));
        if den == 0 || !den.is_finite() {
            break;
        }
        let step = Float::with_val(work, &num / &den);
        if !step.is_finite() {
            break;
        }
        let next = Float::with_val(work, &w - &step);
        w = if next < minus_one {
            // Never leave the principal branch: the midpoint towards −1 keeps
            // the iterate admissible without stalling.
            Float::with_val(work, &w + &minus_one) / 2u32
        } else {
            next
        };
        if step.is_zero() {
            break;
        }
    }

    // `g` evaluated outward, so `hi`/`lo` of the result are rigorous.
    let g = |v: &Float| -> ArbBall {
        let vb = ArbBall::from_point(v, work);
        vb.clone() * vb.exp()
    };
    let xb = ArbBall::from_point(x, work);
    let (x_lo, x_hi) = (xb.lo(), xb.hi());

    // Widen from a plausible accuracy until each side is certified.  The
    // starting step is the working-precision ulp of the guess; 400 doublings
    // reach any representable magnitude.
    let mut unit = Float::with_val(work, w.clone().abs() + 1u32);
    unit >>= work.saturating_sub(4);

    let mut low: Option<Float> = None;
    let mut delta = unit.clone();
    for _ in 0..400 {
        let mut v = Float::with_val(work, &w - &delta);
        if v < minus_one {
            v = minus_one.clone();
        }
        if g(&v).hi() <= x_lo {
            low = Some(v);
            break;
        }
        if v == minus_one {
            break;
        }
        delta *= 2u32;
    }
    // `W₀ ≥ −1` on the whole principal branch, so `−1` is available as a
    // fallback — but *only* once `g(−1) = −1/e ≤ x` has been proved, which is
    // exactly the domain condition.  Accepting `−1` without that check would
    // hand back a bracket for a value that does not exist.
    let low = match low {
        Some(v) => v,
        None if g(&minus_one).hi() <= x_lo => minus_one.clone(),
        None => return None,
    };

    let mut high: Option<Float> = None;
    let mut delta = unit;
    for _ in 0..400 {
        let u = Float::with_val(work, &w + &delta);
        if u >= minus_one && g(&u).lo() >= x_hi {
            high = Some(u);
            break;
        }
        delta *= 2u32;
    }
    let high = high?;
    Some((
        Float::with_val(prec, ArbBall::from_point(&low, prec).lo()),
        Float::with_val(prec, ArbBall::from_point(&high, prec).hi()),
    ))
}

// ---------------------------------------------------------------------------
// AcbBall — complex ball (re ± r_re) + i(im ± r_im)
// ---------------------------------------------------------------------------

/// A complex number represented as two real balls.
///
/// Corresponds to `acb_t` in Arb / FLINT 3.x.
#[derive(Clone, Debug)]
pub struct AcbBall {
    pub re: ArbBall,
    pub im: ArbBall,
}

impl AcbBall {
    pub fn from_real(re: ArbBall) -> Self {
        let prec = re.prec;
        AcbBall {
            re,
            im: ArbBall::new(prec),
        }
    }

    pub fn from_f64(re: f64, im: f64, prec: u32) -> Self {
        AcbBall {
            re: ArbBall::from_f64(re, prec),
            im: ArbBall::from_f64(im, prec),
        }
    }

    pub fn modulus(&self) -> ArbBall {
        // |z| = sqrt(re² + im²)
        let re2 = self.re.clone() * self.re.clone();
        let im2 = self.im.clone() * self.im.clone();
        let sum = re2 + im2;
        sum.sqrt()
            .unwrap_or_else(|| ArbBall::infinity(self.re.prec))
    }
}

impl fmt::Display for AcbBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}·i", self.re, self.im)
    }
}

// ---------------------------------------------------------------------------
// IntervalEval — expression evaluator using ArbBall
// ---------------------------------------------------------------------------

/// Lazily-initialised, process-wide [`PrimitiveRegistry`] used to evaluate
/// `ExprData::Func` nodes.
///
/// A singleton, because building it is far too much work to repeat per `Func`
/// node; the same pattern is used by the JIT's tree-walking interpreter
/// (`crate::jit::registry`).
///
/// `dispatch_registry`, not `default_registry`: this path calls `numeric_ball`
/// on the primitive and treats `None` as unsupported, so it never reads a
/// capability bit — and probing 41 primitives across six argument shapes cost
/// ~1.2 ms of one-time work on whichever call touched the registry first,
/// against 0.02 ms steady state. That is invisible to a wall-clock test and
/// very visible to an instruction-counting one, which is how CodSpeed caught it
/// as a 24x regression on `test_ball_sin_cos_eps1e2`.
///
/// The anti-drift guarantee is unchanged, and in fact stronger: the set of
/// functions accepted here is the set whose `numeric_ball` kernel returns
/// `Some`, which is ground truth rather than a probed summary of it.
fn registry() -> &'static PrimitiveRegistry {
    static REGISTRY: OnceLock<PrimitiveRegistry> = OnceLock::new();
    REGISTRY.get_or_init(PrimitiveRegistry::dispatch_registry)
}

/// Evaluates a symbolic expression using rigorous ball arithmetic.
///
/// Each variable can be bound to an `ArbBall` interval.  The result is an
/// `ArbBall` that is guaranteed to contain the true function value for all
/// inputs in the given intervals.
pub struct IntervalEval {
    bindings: HashMap<ExprId, ArbBall>,
    pub prec: u32,
}

impl IntervalEval {
    pub fn new(prec: u32) -> Self {
        IntervalEval {
            bindings: HashMap::new(),
            prec,
        }
    }

    /// Bind symbol `var` to the ball `ball`.
    pub fn bind(&mut self, var: ExprId, ball: ArbBall) {
        self.bindings.insert(var, ball);
    }

    /// Evaluate `expr` using the current bindings.
    ///
    /// Returns `None` if a node cannot be evaluated (e.g. division by zero,
    /// log of a non-positive ball, unbound variable).
    pub fn eval(&self, expr: ExprId, pool: &ExprPool) -> Option<ArbBall> {
        self.eval_node(expr, pool)
    }

    /// Evaluate a predicate only when its truth value is uniform throughout
    /// every bound input ball.  `None` means the predicate changes (or may
    /// change) within an interval, so choosing a `Piecewise` branch would be
    /// unsound.
    fn eval_predicate(&self, pred: ExprId, pool: &ExprPool) -> Option<bool> {
        let ExprData::Predicate { kind, args } = pool.get(pred) else {
            return None;
        };
        match kind {
            PredicateKind::True => Some(true),
            PredicateKind::False => Some(false),
            PredicateKind::Not => Some(!self.eval_predicate(*args.first()?, pool)?),
            PredicateKind::And => {
                for &a in &args {
                    if !self.eval_predicate(a, pool)? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            PredicateKind::Or => {
                for &a in &args {
                    if self.eval_predicate(a, pool)? {
                        return Some(true);
                    }
                }
                Some(false)
            }
            PredicateKind::Lt
            | PredicateKind::Le
            | PredicateKind::Gt
            | PredicateKind::Ge
            | PredicateKind::Eq
            | PredicateKind::Ne => {
                let [lhs, rhs] = args.as_slice() else {
                    return None;
                };
                let lhs = self.eval_node(*lhs, pool)?;
                let rhs = self.eval_node(*rhs, pool)?;
                let lhs_lo = lhs.lo();
                let lhs_hi = lhs.hi();
                let rhs_lo = rhs.lo();
                let rhs_hi = rhs.hi();

                match kind {
                    // A strict ordering is uniform only when the two closed
                    // balls are separated.  In particular, do not inspect
                    // their midpoints: a midpoint can select the wrong
                    // Piecewise branch when an interval crosses a threshold.
                    PredicateKind::Lt if lhs_hi < rhs_lo => Some(true),
                    PredicateKind::Lt if lhs_lo >= rhs_hi => Some(false),
                    PredicateKind::Le if lhs_hi <= rhs_lo => Some(true),
                    PredicateKind::Le if lhs_lo > rhs_hi => Some(false),
                    PredicateKind::Gt if lhs_lo > rhs_hi => Some(true),
                    PredicateKind::Gt if lhs_hi <= rhs_lo => Some(false),
                    PredicateKind::Ge if lhs_lo >= rhs_hi => Some(true),
                    PredicateKind::Ge if lhs_hi < rhs_lo => Some(false),
                    // Equality is uniformly true only for the same singleton
                    // interval.  It is uniformly false for disjoint balls.
                    PredicateKind::Eq if lhs.is_exact() && rhs.is_exact() && lhs.mid == rhs.mid => {
                        Some(true)
                    }
                    PredicateKind::Eq if lhs_hi < rhs_lo || lhs_lo > rhs_hi => Some(false),
                    // Inequality is the converse only in the cases we can
                    // prove uniformly; overlap remains indeterminate.
                    PredicateKind::Ne if lhs_hi < rhs_lo || lhs_lo > rhs_hi => Some(true),
                    PredicateKind::Ne if lhs.is_exact() && rhs.is_exact() && lhs.mid == rhs.mid => {
                        Some(false)
                    }
                    _ => None,
                }
            }
        }
    }

    fn eval_node(&self, expr: ExprId, pool: &ExprPool) -> Option<ArbBall> {
        match pool.get(expr) {
            ExprData::Integer(n) => Some(ArbBall::from_integer(&n.0, self.prec)),
            ExprData::Rational(r) => Some(ArbBall::from_rational(&r.0, self.prec)),
            ExprData::Float(f) => Some(ArbBall::from_f64(f.inner.to_f64(), self.prec)),
            ExprData::Symbol { .. } => self.bindings.get(&expr).cloned(),
            ExprData::Add(args) => {
                let mut acc = ArbBall::from_f64(0.0, self.prec);
                for &a in &args {
                    acc = acc + self.eval_node(a, pool)?;
                }
                Some(acc)
            }
            ExprData::Mul(args) => {
                let mut acc = ArbBall::from_f64(1.0, self.prec);
                for &a in &args {
                    acc = acc * self.eval_node(a, pool)?;
                }
                Some(acc)
            }
            ExprData::Pow { base, exp } => {
                let b = self.eval_node(base, pool)?;
                let e = self.eval_node(exp, pool)?;
                // Integer exponent path for exact results
                if let ExprData::Integer(n) = pool.get(exp) {
                    let nv = n.0.to_i64()?;
                    return Some(b.powi(nv));
                }
                Some(b.pow_f(&e))
            }
            // Every `Func` node is dispatched through the primitive registry's
            // `numeric_ball` slot, so the set of functions interval evaluation
            // accepts *is* the set the registry advertises as
            // `Capabilities::NUMERIC_BALL` — by construction, not by
            // agreement. The hand-written match this replaces had drifted:
            // `bessel_j0`/`bessel_j1` had outward-rounded ball kernels and a
            // `numeric_ball: true` capability bit, and were still refused here
            // with `E-EVAL-010` for no reason anyone had recorded.
            //
            // Arity is the primitive's own business: each kernel declines an
            // argument list it does not have a rule for (see `builtins::unary`),
            // which is what makes dispatching the whole list safe.
            ExprData::Func { name, args } if !args.is_empty() => {
                let mut vals = Vec::with_capacity(args.len());
                for &a in &args {
                    vals.push(self.eval_node(a, pool)?);
                }
                registry().numeric_ball(&name, &vals)
            }
            ExprData::Piecewise { branches, default } => {
                for (c, v) in branches {
                    match self.eval_predicate(c, pool) {
                        Some(true) => return self.eval_node(v, pool),
                        Some(false) => {}
                        None => return None,
                    }
                }
                self.eval_node(default, pool)
            }
            ExprData::Predicate { .. } => {
                let v = if self.eval_predicate(expr, pool)? {
                    1.0
                } else {
                    0.0
                };
                Some(ArbBall::from_f64(v, self.prec))
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn ball_contains_midpoint() {
        let b = ArbBall::from_midpoint_radius(3.0, 0.5, 64);
        assert!(b.contains(3.0));
        assert!(b.contains(2.5));
        assert!(b.contains(3.5));
        assert!(!b.contains(4.0));
    }

    #[test]
    fn ball_add_enclosure() {
        let a = ArbBall::from_midpoint_radius(1.0, 0.1, 64);
        let b = ArbBall::from_midpoint_radius(2.0, 0.2, 64);
        let c = a + b;
        // True result: [2.7, 3.3]
        assert!(c.contains(2.7));
        assert!(c.contains(3.0));
        assert!(c.contains(3.3));
    }

    #[test]
    fn ball_mul_enclosure() {
        let a = ArbBall::from_midpoint_radius(2.0, 0.5, 64); // [1.5, 2.5]
        let b = ArbBall::from_midpoint_radius(3.0, 0.5, 64); // [2.5, 3.5]
        let c = a * b;
        // True range: [1.5*2.5, 2.5*3.5] = [3.75, 8.75]
        assert!(c.contains(4.0));
        assert!(c.contains(8.0));
    }

    #[test]
    fn ball_lambert_w0_at_one() {
        let b = ArbBall::from_f64(1.0, 128);
        assert!(b.lambert_w0().is_some());
    }

    #[test]
    fn ball_powi_exact() {
        let b = ArbBall::from_f64(3.0, 128);
        let b3 = b.powi(3);
        assert!(b3.contains(27.0));
        assert!(!b3.contains(26.0));
    }

    #[test]
    fn ball_sin_enclosure() {
        // sin(π/2) = 1
        let pi_2 = std::f64::consts::PI / 2.0;
        let b = ArbBall::from_midpoint_radius(pi_2, 0.01, 128);
        let s = b.sin();
        assert!(s.contains(1.0));
    }

    #[test]
    fn ball_exp_enclosure() {
        let b = ArbBall::from_midpoint_radius(0.0, 0.1, 128); // [-0.1, 0.1]
        let e = b.exp();
        // e^{-0.1} ≈ 0.905, e^{0.1} ≈ 1.105
        assert!(e.contains(0.905));
        assert!(e.contains(1.0));
        assert!(e.contains(1.105));
    }

    #[test]
    fn ball_log_enclosure() {
        let b = ArbBall::from_midpoint_radius(2.0, 0.5, 128); // [1.5, 2.5]
        let l = b.log().unwrap();
        // ln(1.5) ≈ 0.40547, ln(2.5) ≈ 0.91629 — use values safely inside
        assert!(l.contains(0.41));
        assert!(l.contains(0.91));
        // midpoint ln(2) ≈ 0.6931 must be contained
        assert!(l.contains(2_f64.ln()));
    }

    #[test]
    fn ball_log_fails_at_nonpositive() {
        let b = ArbBall::from_midpoint_radius(0.0, 0.5, 128); // contains negative
        assert!(b.log().is_none());
    }

    #[test]
    fn interval_eval_constant() {
        let pool = p();
        let five = pool.integer(5_i32);
        let eval = IntervalEval::new(128);
        let r = eval.eval(five, &pool).unwrap();
        assert!(r.contains(5.0));
    }

    #[test]
    fn interval_eval_piecewise_with_binding() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_gt(x, pool.integer(0_i32)), x)],
            pool.integer(-1_i32),
        );
        let mut ev = IntervalEval::new(128);
        ev.bind(x, ArbBall::from_midpoint_radius(1.0, 1e-6, 128));
        let r = ev.eval(pw, &pool).unwrap();
        assert!(r.contains(1.0));
    }

    #[test]
    fn interval_eval_refuses_piecewise_threshold_spanning_binding() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let pw = pool.piecewise(
            vec![(pool.pred_gt(x, pool.integer(0_i32)), pool.integer(1_i32))],
            pool.integer(-1_i32),
        );
        let mut ev = IntervalEval::new(128);
        ev.bind(x, ArbBall::from_midpoint_radius(0.1, 0.2, 128));

        // The interval contains both negative and positive values.  Selecting
        // the positive branch from the midpoint would exclude -1.
        assert!(ev.eval(pw, &pool).is_none());
    }

    #[test]
    fn interval_eval_polynomial() {
        // f(x) = x² + 1,  x ∈ [2.9, 3.1]
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let one = pool.integer(1_i32);
        let expr = pool.add(vec![x2, one]);

        let x_ball = ArbBall::from_midpoint_radius(3.0, 0.1, 128);
        let mut eval = IntervalEval::new(128);
        eval.bind(x, x_ball);
        let r = eval.eval(expr, &pool).unwrap();
        // f([2.9, 3.1]) ⊆ [2.9² + 1, 3.1² + 1] = [9.41, 10.61]
        assert!(r.contains(9.5));
        assert!(r.contains(10.0));
        assert!(r.contains(10.5));
    }

    #[test]
    fn interval_eval_unbound_is_none() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let eval = IntervalEval::new(128);
        assert!(eval.eval(x, &pool).is_none());
    }

    #[test]
    fn interval_eval_rational() {
        let pool = p();
        let third = pool.rational(1, 3);
        let eval = IntervalEval::new(128);
        let r = eval.eval(third, &pool).unwrap();
        // 1/3 ≈ 0.3333...; check mid_f64 is close to 1/3 and ball is tiny
        let mid = r.mid_f64();
        assert!((mid - 1.0 / 3.0).abs() < 1e-15, "mid={mid}");
        // Radius should be very small (< 1 ulp at double precision ≈ 1.5e-17)
        assert!(r.rad_f64() < 1e-30, "rad={}", r.rad_f64());
    }

    /// The dispatch is *derived* from the registry, so this cannot drift the
    /// way the hand-written match did. Pin it anyway: the property that used to
    /// be violated (a primitive advertising `numeric_ball` that interval
    /// evaluation refuses) is the whole point, and a future refactor that
    /// reintroduces a list here fails this test rather than a user's proof.
    #[test]
    fn interval_eval_accepts_every_primitive_that_advertises_numeric_ball() {
        use crate::primitive::{Capabilities, PrimitiveRegistry};
        let reg = PrimitiveRegistry::default_registry();
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // No single probe point is in every kernel's domain — `acosh` needs
        // `x ≥ 1` and `atanh` needs `|x| < 1` — so a primitive counts as
        // reachable if *some* probe evaluates. A domain refusal is a different
        // answer from "I have no rule for this name", and only the second is
        // the coverage gap under test.
        let probes = [1.5_f64, 0.5];
        let refused: Vec<String> = reg
            .iter()
            .filter(|(_, caps)| caps.contains(Capabilities::NUMERIC_BALL))
            .filter(|(name, _)| {
                let call = pool.func(*name, vec![x]);
                probes.iter().all(|&probe| {
                    let mut ev = IntervalEval::new(128);
                    ev.bind(x, ArbBall::from_f64(probe, 128));
                    ev.eval(call, &pool).is_none()
                })
            })
            .map(|(name, _)| name.to_string())
            .collect();
        assert!(
            refused.is_empty(),
            "these primitives advertise numeric_ball but interval evaluation \
             refuses them: {refused:?}"
        );
    }

    /// The two the audit was opened for, pinned by name and by value.
    #[test]
    fn interval_eval_evaluates_bessel() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let mut ev = IntervalEval::new(128);
        ev.bind(x, ArbBall::from_f64(1.0, 128));
        let j0 = ev.eval(pool.func("bessel_j0", vec![x]), &pool).unwrap();
        let j1 = ev.eval(pool.func("bessel_j1", vec![x]), &pool).unwrap();
        // The enclosure at an exact point is far tighter than an `f64` literal,
        // so compare midpoints rather than asking it to `contain` one.
        assert!(
            (j0.mid_f64() - 0.765_197_686_557_966_5).abs() < 1e-15,
            "J0(1) = {j0}"
        );
        assert!(
            (j1.mid_f64() - 0.440_050_585_744_933_5).abs() < 1e-15,
            "J1(1) = {j1}"
        );
        assert!(j0.rad_f64() < 1e-30 && j1.rad_f64() < 1e-30);
    }

    /// A unary kernel handed the wrong number of arguments must decline, not
    /// quietly bound the first one: a rigorous enclosure of the wrong function
    /// is the worst answer this module can give.
    #[test]
    fn interval_eval_refuses_a_unary_primitive_at_the_wrong_arity() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut ev = IntervalEval::new(128);
        ev.bind(x, ArbBall::from_f64(1.0, 128));
        ev.bind(y, ArbBall::from_f64(2.0, 128));
        assert!(ev.eval(pool.func("sin", vec![x, y]), &pool).is_none());
        assert!(ev.eval(pool.func("sin", vec![]), &pool).is_none());
    }

    #[test]
    fn acb_modulus() {
        // |3 + 4i| = 5
        let z = AcbBall::from_f64(3.0, 4.0, 128);
        let m = z.modulus();
        assert!(m.contains(5.0));
    }
}

#[cfg(test)]
mod rounding_soundness_tests {
    use super::*;

    const PREC: u32 = 128;

    /// Containment tested at ball precision.
    ///
    /// `ArbBall::contains` takes an `f64`, and rounding the true value to
    /// `f64` moves it by ~2^-53 — enough to push a sample that sits exactly on
    /// an endpoint outside an enclosure whose outward rounding is 2^-126 wide.
    /// That is a defect of the *measurement*, not of the enclosure.
    fn encloses(b: &ArbBall, v: &Float) -> bool {
        *v >= b.lo() && *v <= b.hi()
    }

    /// Every transcendental op must carry a rounding term.
    ///
    /// `exp`/`log`/`sqrt`/`tan`/`asin`/`acos`/`atan`/`asinh`/`atanh` built their
    /// result from `f(lo)` and `f(hi)` at finite precision and took the spread
    /// as the radius. Given an *exact* input those endpoints coincide, so the
    /// radius came out exactly zero — claiming a transcendental value is
    /// exactly representable. `sin`/`cos` already added the term.
    #[test]
    fn transcendental_ops_never_report_zero_radius_on_exact_input() {
        let half = ArbBall::from_f64(0.5, PREC);
        let two = ArbBall::from_f64(2.0, PREC);

        let mut zero_radius: Vec<&str> = Vec::new();
        let mut check = |name: &'static str, b: ArbBall| {
            if b.rad == 0 {
                zero_radius.push(name);
            }
        };

        check("exp", half.exp());
        check("sin", half.sin());
        check("cos", half.cos());
        check("log", two.log().expect("log 2 defined"));
        check("sqrt", two.sqrt().expect("sqrt 2 defined"));
        check("tan", half.tan().expect("tan 0.5 defined"));
        check("asin", half.asin().expect("asin 0.5 defined"));
        check("acos", half.acos().expect("acos 0.5 defined"));
        check("atan", half.atan());
        check("asinh", half.asinh());
        check("atanh", half.atanh().expect("atanh 0.5 defined"));

        assert!(
            zero_radius.is_empty(),
            "these ops claim an irrational result is exact: {zero_radius:?}"
        );
    }

    /// Jₙ oscillates, so the endpoint hull `bessel_jn` used to take was not an
    /// enclosure: on `[-1, 1]` both endpoints give `J₀(±1) ≈ 0.7652`, the hull
    /// collapsed to that point, and `J₀(0) = 1` — the maximum of the function —
    /// fell outside the "rigorous" ball.
    #[test]
    fn bessel_encloses_an_interior_extremum() {
        let b = ArbBall::from_midpoint_radius(0.0, 1.0, PREC); // [-1, 1]
        let j0 = b.bessel_jn(0);
        assert!(j0.contains(1.0), "J0 on [-1,1] misses its maximum: {j0}");
        assert!(j0.contains(0.7651976865579666), "{j0}");

        // Same failure one period out, where the endpoints straddle a zero
        // rather than a peak.
        let b = ArbBall::from_midpoint_radius(3.0, 1.0, PREC); // [2, 4]
        let j1 = b.bessel_jn(1);
        for x in [2.0_f64, 2.5, 3.0, 3.5, 4.0] {
            let mut v = Float::with_val(PREC, x);
            v.jn_mut(1);
            assert!(j1.contains(v.to_f64()), "J1({x}) outside {j1}");
        }
    }

    /// Randomised enclosure check: sampling the true function inside the ball
    /// must never escape the reported enclosure.
    #[test]
    fn bessel_enclosure_holds_on_random_intervals() {
        let mut seed = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let centre = (next() - 0.5) * 40.0;
            let radius = next() * 5.0;
            for n in [0_i32, 1] {
                let ball = ArbBall::from_midpoint_radius(centre, radius, PREC);
                let out = ball.bessel_jn(n);
                for k in 0..=20 {
                    let x = centre - radius + 2.0 * radius * (k as f64 / 20.0);
                    let mut v = Float::with_val(PREC, x);
                    v.jn_mut(n);
                    assert!(
                        out.contains(v.to_f64()),
                        "J{n}({x}) = {v} escapes {out} for [{}, {}]",
                        centre - radius,
                        centre + radius
                    );
                }
            }
        }
    }

    /// `tan` used to hull its two endpoints whenever they came out in
    /// increasing order, which a box crossing a pole by nearly a whole period
    /// does: `[0.1, 0.1 + π]` was reported as `[0.1003…, 0.2027…]`, and
    /// `tan(1.5) = 14.101…` sits inside that box.
    #[test]
    fn tan_encloses_a_box_that_crosses_a_pole() {
        let pi = std::f64::consts::PI;
        for (lo, hi) in [
            (0.1, 0.2 + pi),
            (0.1, 0.5 + pi),
            (-1.0, 1.0 + pi),
            (-1.4292036732051034, 4.570796326794897),
            (1.5, 1.7),    // straddles π/2 by less than a period
            (-10.0, 10.0), // several periods
        ] {
            let b = ArbBall::from_midpoint_radius((lo + hi) / 2.0, (hi - lo) / 2.0, PREC);
            let out = b.tan().expect("finite box");
            // Sample the true function across the box; every value must be in.
            for k in 0..=200 {
                let x = lo + (hi - lo) * (k as f64 / 200.0);
                let v = Float::with_val(PREC + 32, x).tan();
                assert!(
                    encloses(&out, &v),
                    "tan({x}) = {v} escapes {out} for [{lo}, {hi}]"
                );
            }
        }
    }

    /// …and the fix must not answer `∞` to everything: a pole-free box still
    /// gets the exact endpoint hull, which is what `tan′ = sec² > 0` buys.
    #[test]
    fn tan_stays_tight_on_a_pole_free_box() {
        let b = ArbBall::from_midpoint_radius(0.55, 0.45, PREC); // [0.1, 1.0]
        let out = b.tan().expect("finite box");
        assert!(out.lo() > 0.10033, "lower bound {} is not tight", out.lo());
        assert!(out.hi() < 1.5575, "upper bound {} is not tight", out.hi());
        assert!(out.contains(0.5463024898437905)); // tan(0.5)
    }

    /// `pow_f` hulled the four corners of `[lo_b, hi_b] × [lo_e, hi_e]`, which
    /// encloses the range only where `x ↦ x^y` is monotone in `x`. A base
    /// straddling zero with an even exponent is the counterexample: `x²` over
    /// `[-1, 3]` came out as `[1, 9]`, missing `x = 0 ↦ 0`. With a *negative*
    /// exponent the claim was worse than narrow — `[0.111…, 1]` for a function
    /// unbounded on the box.
    #[test]
    fn pow_f_encloses_a_base_straddling_zero() {
        let base = ArbBall::from_midpoint_radius(1.0, 2.0, PREC); // [-1, 3]
        for e in [2.0_f64, 4.0, 2.0, 6.0] {
            let out = base.pow_f(&ArbBall::from_f64(e, PREC));
            for k in 0..=100 {
                let x = -1.0 + 4.0 * (k as f64 / 100.0);
                let v = Float::with_val(PREC + 32, x).pow(e);
                assert!(encloses(&out, &v), "{x}^{e} = {v} escapes {out}");
            }
        }
        // A negative exponent puts a pole at 0 inside the box.
        let out = base.pow_f(&ArbBall::from_f64(-2.0, PREC));
        assert!(
            out.contains(1e6),
            "x^-2 over [-1, 3] claims the finite {out}"
        );
        assert!(
            out.contains(1e300),
            "x^-2 over [-1, 3] claims the finite {out}"
        );
        // A non-integer exponent on a negative base is not real at all.
        let out = base.pow_f(&ArbBall::from_f64(1.5, PREC));
        assert!(out.rad.is_infinite(), "(-1)^1.5 is not real, got {out}");
    }

    /// The corner hull is kept where it is valid — a base that stays positive —
    /// so this must not have widened into uselessness.
    #[test]
    fn pow_f_stays_tight_on_a_positive_base() {
        let base = ArbBall::from_midpoint_radius(2.5, 0.5, PREC); // [2, 3]
        let out = base.pow_f(&ArbBall::from_f64(0.5, PREC));
        let (sqrt2, sqrt3) = (std::f64::consts::SQRT_2, 1.7320508075688772_f64);
        assert!(
            out.lo() > sqrt2 - 1e-9,
            "lower bound {} is not tight",
            out.lo()
        );
        assert!(
            out.hi() < sqrt3 + 1e-9,
            "upper bound {} is not tight",
            out.hi()
        );
        // Exponent intervals are allowed too: x^y over [2,3] × [-1, 2].
        let out = base.pow_f(&ArbBall::from_midpoint_radius(0.5, 1.5, PREC));
        for k in 0..=20 {
            let x = 2.0 + (k as f64) / 20.0;
            for j in 0..=20 {
                let y = -1.0 + 3.0 * (j as f64) / 20.0;
                let v = Float::with_val(PREC + 32, x).pow(y);
                assert!(encloses(&out, &v), "{x}^{y} = {v} escapes {out}");
            }
        }
        assert!(out.hi() < 9.000001, "upper bound {} is not tight", out.hi());
    }

    /// Randomised enclosure check over *wide* balls for every kernel that
    /// builds its answer from the two endpoints.  This is the sweep that would
    /// have caught `tan` and `pow_f`; a point ball reaches neither.
    ///
    /// Samples are taken from the ball's *own* endpoints in `Float` arithmetic:
    /// recomputing `mid ± rad` in `f64` moves the sample by ~2^-53, which is
    /// 10¹⁸ times the outward rounding an enclosure carries, and would put the
    /// end samples outside the box being tested rather than on its boundary.
    #[test]
    fn endpoint_hull_kernels_enclose_random_wide_intervals() {
        let work = PREC + 32;
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..200 {
            let centre = (next() - 0.5) * 20.0;
            let radius = next() * 5.0;
            let ball = ArbBall::from_midpoint_radius(centre, radius, PREC);
            let lo = ball.lo();
            let width = Float::with_val(work, ball.hi() - &lo);
            let kernels: Vec<(&str, Option<ArbBall>)> = vec![
                ("exp", Some(ball.exp())),
                ("sin", Some(ball.sin())),
                ("cos", Some(ball.cos())),
                ("tan", ball.tan()),
                ("log", ball.log()),
                ("sqrt", ball.sqrt()),
                ("sinh", Some(ball.sinh())),
                ("cosh", Some(ball.cosh())),
                ("tanh", Some(ball.tanh())),
                ("asin", ball.asin()),
                ("acos", ball.acos()),
                ("atan", Some(ball.atan())),
                ("asinh", Some(ball.asinh())),
                ("acosh", ball.acosh()),
                ("atanh", ball.atanh()),
                ("erf", Some(ball.erf())),
                ("erfc", Some(ball.erfc())),
                ("gamma", ball.gamma()),
                ("digamma", ball.digamma()),
                ("bessel_j0", Some(ball.bessel_jn(0))),
                ("square", Some(ball.pow_f(&ArbBall::from_f64(2.0, PREC)))),
                ("cube", Some(ball.pow_f(&ArbBall::from_f64(3.0, PREC)))),
                ("recip_sq", Some(ball.pow_f(&ArbBall::from_f64(-2.0, PREC)))),
                ("sqrt_pow", Some(ball.pow_f(&ArbBall::from_f64(0.5, PREC)))),
            ];
            for (name, out) in kernels {
                let Some(out) = out else { continue };
                for k in 0..=50 {
                    let t = Float::with_val(work, k as f64 / 50.0);
                    let x = Float::with_val(work, &lo + Float::with_val(work, &width * &t));
                    let v = match name {
                        "exp" => x.clone().exp(),
                        "sin" => x.clone().sin(),
                        "cos" => x.clone().cos(),
                        "tan" => x.clone().tan(),
                        "log" => x.clone().ln(),
                        "sqrt" => x.clone().sqrt(),
                        "sinh" => x.clone().sinh(),
                        "cosh" => x.clone().cosh(),
                        "tanh" => x.clone().tanh(),
                        "asin" => x.clone().asin(),
                        "acos" => x.clone().acos(),
                        "atan" => x.clone().atan(),
                        "asinh" => x.clone().asinh(),
                        "acosh" => x.clone().acosh(),
                        "atanh" => x.clone().atanh(),
                        "erf" => x.clone().erf(),
                        "erfc" => x.clone().erfc(),
                        "gamma" => x.clone().gamma(),
                        "digamma" => {
                            let mut t = x.clone();
                            t.digamma_mut();
                            t
                        }
                        "bessel_j0" => {
                            let mut t = x.clone();
                            t.jn_mut(0);
                            t
                        }
                        "square" => Float::with_val(work, &x * &x),
                        "cube" => Float::with_val(work, Float::with_val(work, &x * &x) * &x),
                        "recip_sq" => Float::with_val(work, 1u32 / Float::with_val(work, &x * &x)),
                        "sqrt_pow" => x.clone().sqrt(),
                        _ => unreachable!(),
                    };
                    if !v.is_finite() {
                        continue;
                    }
                    assert!(
                        encloses(&out, &v),
                        "{name}({x}) = {v} escapes {out} on [{lo}, {}]",
                        ball.hi()
                    );
                }
            }
        }
    }

    /// `digamma` used to reject its pole by walking *every* integer in the box,
    /// which is 10¹⁸ iterations on a box this wide.
    #[test]
    fn digamma_does_not_walk_every_integer_in_the_box() {
        // [1e16, 9.9e17] — an f64-exact box with 10^18 integers in it.
        let b = ArbBall::from_midpoint_radius(5e17, 4.9e17, PREC);
        let out = b.digamma().expect("no pole in [1e16, 9.9e17]");
        for x in [1e16_f64, 5e17, 9.9e17] {
            let mut v = Float::with_val(PREC + 32, x);
            v.digamma_mut();
            assert!(encloses(&out, &v), "ψ({x}) = {v} escapes {out}");
        }
        // The pole test still fires when a non-positive integer is in reach.
        assert!(ArbBall::from_midpoint_radius(0.0, 1.0, PREC)
            .digamma()
            .is_none());
        assert!(ArbBall::from_midpoint_radius(-2.5, 1.0, PREC)
            .digamma()
            .is_none());
    }

    /// The radius must stay at the working-precision scale, not balloon.
    ///
    /// Soundness is trivially achievable by making every ball enormous; this
    /// pins that the fix did not take that route.
    #[test]
    fn rounding_term_stays_at_precision_scale() {
        let b = ArbBall::from_f64(0.5, PREC).exp();
        let bound = Float::with_val(PREC, 1e-30);
        assert!(
            b.rad < bound,
            "radius {} is far above the 2^-prec scale",
            b.rad
        );
    }
}

#[cfg(test)]
mod special_kernel_tests {
    use super::*;

    const P: u32 = 128;

    /// `W₀` at a point, against a value known to more digits than the ball
    /// claims.  This is the failure the certified bracket replaces: the old
    /// `f64` kernel returned a radius of 10⁻³⁹ around a midpoint 3·10⁻¹⁷ away
    /// from the truth, so the enclosure excluded its own value.
    #[test]
    fn lambert_w0_encloses_high_precision_values() {
        // W₀(1) = Ω, the omega constant, to 40 digits.
        let omega = Float::with_val(
            256,
            Float::parse("0.5671432904097838729999686622103555497538").unwrap(),
        );
        let b = ArbBall::from_f64(1.0, P).lambert_w0().unwrap();
        assert!(
            b.lo() <= omega && omega <= b.hi(),
            "W₀(1) = {omega} escaped [{}, {}]",
            b.lo(),
            b.hi()
        );
        assert!(b.rad_f64() < 1e-30, "rad = {}", b.rad_f64());

        // W₀(e) = 1 exactly.
        let e = Float::with_val(P, 1u32).exp();
        let b = ArbBall {
            mid: e,
            rad: Float::new(P),
            prec: P,
        }
        .lambert_w0()
        .unwrap();
        assert!(b.contains(1.0), "W₀(e) must enclose 1, got {b}");
        assert!(b.rad_f64() < 1e-30);
    }

    /// `W₀(x)·e^{W₀(x)} = x` checked *through the enclosure*: every point of
    /// the returned ball is a candidate, so the identity has to hold for the
    /// ball as a whole.
    #[test]
    fn lambert_w0_enclosure_satisfies_its_defining_equation() {
        for x in [-0.3, -0.1, 0.0, 0.25, 1.0, 2.5, 10.0, 1e3, 1e10] {
            let xb = ArbBall::from_f64(x, P);
            let w = xb.lambert_w0().unwrap_or_else(|| panic!("W₀({x}) refused"));
            let g = w.clone() * w.exp();
            assert!(g.contains(x), "W₀({x}) enclosure gives g = {g}, not {x}");
        }
    }

    /// Off-domain arguments refuse rather than answer.
    #[test]
    fn lambert_w0_refuses_below_the_branch_point() {
        for x in [-0.5, -0.4, -0.37, -1.0, -1e6] {
            assert!(
                ArbBall::from_f64(x, P).lambert_w0().is_none(),
                "W₀({x}) is not real but was answered"
            );
        }
    }

    /// Over a genuine box, and monotonically: `W₀` increases, so the enclosure
    /// must bracket both endpoint values and everything between.
    #[test]
    fn lambert_w0_over_a_box_brackets_interior_points() {
        let b = ArbBall::from_midpoint_radius(1.5, 1.0, P)
            .lambert_w0()
            .unwrap();
        for k in 0..=50 {
            let x = 0.5 + 2.0 * (k as f64) / 50.0;
            let w = crate::special::lambert_w0(x).unwrap();
            assert!(
                b.lo().to_f64() - 1e-12 <= w && w <= b.hi().to_f64() + 1e-12,
                "W₀({x}) = {w} escaped {b}"
            );
        }
    }

    /// Γ over boxes, including one straddling the minimum at x ≈ 1.4616 where
    /// an endpoint hull would be wrong in the same way `bessel_jn`'s was.
    #[test]
    fn gamma_brackets_dense_samples() {
        for (lo, hi) in [
            (0.5_f64, 0.6_f64),
            (1.0, 2.0),
            (1.4, 1.5),
            (0.25, 3.0),
            (4.0, 4.25),
            (9.0, 10.0),
            (0.01, 0.02),
        ] {
            let b = ArbBall::from_endpoints(&Float::with_val(P, lo), &Float::with_val(P, hi), P)
                .gamma()
                .unwrap_or_else(|| panic!("Γ on [{lo},{hi}] refused"));
            for k in 0..=100 {
                let t = lo + (hi - lo) * (k as f64) / 100.0;
                let truth = Float::with_val(P + 64, t).gamma();
                assert!(
                    b.lo() <= truth && truth <= b.hi(),
                    "Γ({t}) = {truth} escaped [{}, {}]",
                    b.lo(),
                    b.hi()
                );
            }
        }
    }

    /// The minimum of Γ sits *inside* [1, 2]: max(Γ(1), Γ(2)) = 1 is the top
    /// of the range, and a hull of the endpoints would collapse to the single
    /// point 1 and miss Γ(1.4616) = 0.8856.
    #[test]
    fn gamma_does_not_assume_monotonicity() {
        let b = ArbBall::from_endpoints(&Float::with_val(P, 1.0), &Float::with_val(P, 2.0), P)
            .gamma()
            .unwrap();
        assert!(b.lo() < 0.8857, "lower bound {} misses the minimum", b.lo());
        assert!(b.hi() >= 1.0, "upper bound {} misses Γ(1) = 1", b.hi());
    }

    #[test]
    fn gamma_refuses_non_positive_boxes() {
        for (lo, hi) in [(-1.0_f64, 1.0_f64), (0.0, 1.0), (-3.0, -2.0), (-0.5, -0.4)] {
            assert!(
                ArbBall::from_endpoints(&Float::with_val(P, lo), &Float::with_val(P, hi), P)
                    .gamma()
                    .is_none(),
                "Γ on [{lo},{hi}] should refuse"
            );
        }
    }
}
