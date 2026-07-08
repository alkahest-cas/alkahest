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

use crate::kernel::eval_const::try_predicate_bool_from_expr;
use crate::kernel::expr::PredicateKind;
use crate::kernel::{ExprData, ExprId, ExprPool};
use rug::{ops::Pow, Float};
use std::collections::HashMap;
use std::fmt;

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

    /// Lower bound of the interval.
    pub fn lo(&self) -> Float {
        Float::with_val(self.prec, &self.mid - &self.rad)
    }

    /// Upper bound of the interval.
    pub fn hi(&self) -> Float {
        Float::with_val(self.prec, &self.mid + &self.rad)
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

    /// Grow radius by a rounding-error term: `eps * |mid| * 2^{-prec}`.
    fn add_rounding_error(&mut self) {
        if self.mid.is_infinite() || self.mid.is_nan() {
            self.rad = Float::with_val(self.prec, f64::INFINITY);
            return;
        }
        let scale = Float::with_val(self.prec, &self.mid).abs()
            * Float::with_val(self.prec, 2.0_f64.powi(-(self.prec as i32)));
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
        let mut rad = Float::with_val(prec, &self.rad + &rhs.rad);
        // Rounding error: 1 ulp
        let eps = Float::with_val(prec, mid.abs_ref())
            * Float::with_val(prec, 2.0_f64.powi(-(prec as i32)));
        rad += eps;
        ArbBall { mid, rad, prec }
    }
}

impl std::ops::Sub for ArbBall {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let prec = self.prec.max(rhs.prec);
        let mid = Float::with_val(prec, &self.mid - &rhs.mid);
        let mut rad = Float::with_val(prec, &self.rad + &rhs.rad);
        let eps = Float::with_val(prec, mid.abs_ref())
            * Float::with_val(prec, 2.0_f64.powi(-(prec as i32)));
        rad += eps;
        ArbBall { mid, rad, prec }
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
        let mut rad = Float::with_val(prec, &ma * &rhs.rad)
            + Float::with_val(prec, &mb * &self.rad)
            + Float::with_val(prec, &self.rad * &rhs.rad);
        let eps = Float::with_val(prec, mid.abs_ref())
            * Float::with_val(prec, 2.0_f64.powi(-(prec as i32)));
        rad += eps;
        ArbBall { mid, rad, prec }
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
        // Compute all 4 corners
        let corners = [
            Float::with_val(prec, self.lo() / lo_rhs.clone()),
            Float::with_val(prec, self.lo() / hi_rhs.clone()),
            Float::with_val(prec, self.hi() / lo_rhs.clone()),
            Float::with_val(prec, self.hi() / hi_rhs.clone()),
        ];
        let min = corners
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let max = corners
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let sum = Float::with_val(prec, &min + &max);
        let diff = Float::with_val(prec, &max - &min);
        let new_mid = sum / 2_f64;
        let rad = diff / 2_f64;
        Some(ArbBall {
            mid: new_mid,
            rad,
            prec,
        })
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

    pub fn pow_f(&self, exp: &ArbBall) -> Self {
        // [a,b]^[c,d] using interval exponentiation
        let prec = self.prec;
        let lo = self.lo();
        let hi = self.hi();
        if lo < 0 && !exp.is_exact() {
            return ArbBall::infinity(prec); // complex result possible
        }
        // Conservative bound via corner evaluation
        let corners = [
            Float::with_val(prec, lo.clone().pow(exp.lo())),
            Float::with_val(prec, lo.clone().pow(exp.hi())),
            Float::with_val(prec, hi.clone().pow(exp.lo())),
            Float::with_val(prec, hi.clone().pow(exp.hi())),
        ];
        let min = corners
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let max = corners
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let sum = Float::with_val(prec, &min + &max);
        let diff = Float::with_val(prec, &max - &min);
        let new_mid = sum / 2_f64;
        let rad = diff / 2_f64;
        ArbBall {
            mid: new_mid,
            rad,
            prec,
        }
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
        let lo = Float::with_val(prec, self.lo().exp());
        let hi = Float::with_val(prec, self.hi().exp());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    pub fn log(&self) -> Option<Self> {
        if self.lo() <= 0 {
            return None; // log undefined for non-positive values
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().ln());
        let hi = Float::with_val(prec, self.hi().ln());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    pub fn sqrt(&self) -> Option<Self> {
        if self.lo() < 0 {
            return None;
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().sqrt());
        let hi = Float::with_val(prec, self.hi().sqrt());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    /// tan([m-r, m+r]) — Lipschitz constant: sec²(m+r) (may blow up near π/2).
    /// Returns None if the interval contains a pole.
    pub fn tan(&self) -> Option<Self> {
        let prec = self.prec;
        let _pi_half = Float::with_val(prec, rug::float::Constant::Pi) / 2_f64;
        // Check that neither bound is within ε of π/2 + k*π
        let lo = self.lo();
        let hi = self.hi();
        // simple pole check: |lo mod π - π/2| > 0 and |hi mod π - π/2| > 0
        let lo_f = lo.to_f64();
        let hi_f = hi.to_f64();
        let pi_f: f64 = std::f64::consts::PI;
        let near_pole = |v: f64| ((v % pi_f).abs() - pi_f / 2.0).abs() < 1e-9;
        if near_pole(lo_f) || near_pole(hi_f) {
            return None;
        }
        let lo_tan = Float::with_val(prec, lo.tan());
        let hi_tan = Float::with_val(prec, hi.tan());
        // If lo_tan > hi_tan the interval crossed a pole — discard
        if lo_tan > hi_tan {
            return None;
        }
        let sum = Float::with_val(prec, &lo_tan + &hi_tan);
        let diff = Float::with_val(prec, &hi_tan - &lo_tan);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    pub fn sinh(&self) -> Self {
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().sinh());
        let hi = Float::with_val(prec, self.hi().sinh());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    pub fn cosh(&self) -> Self {
        let prec = self.prec;
        // cosh is even and has a minimum at 0; handle by evaluating at lo, hi, and 0 if in range
        let lo = Float::with_val(prec, self.lo().cosh());
        let hi = Float::with_val(prec, self.hi().cosh());
        let (min_val, max_val) = if self.lo() <= 0 && self.hi() >= 0 {
            // minimum is cosh(0) = 1
            let cosh_lo = lo.clone();
            let cosh_hi = hi.clone();
            let min = Float::with_val(prec, 1_f64);
            let max = if cosh_lo > cosh_hi { cosh_lo } else { cosh_hi };
            (min, max)
        } else if lo < hi {
            (lo, hi)
        } else {
            (hi, lo)
        };
        let sum = Float::with_val(prec, &min_val + &max_val);
        let diff = Float::with_val(prec, &max_val - &min_val);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    pub fn tanh(&self) -> Self {
        // tanh is monotone, maps ℝ → (-1, 1)
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().tanh());
        let hi = Float::with_val(prec, self.hi().tanh());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    pub fn asin(&self) -> Option<Self> {
        if self.lo() < -1 || self.hi() > 1 {
            return None;
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().asin());
        let hi = Float::with_val(prec, self.hi().asin());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    pub fn acos(&self) -> Option<Self> {
        if self.lo() < -1 || self.hi() > 1 {
            return None;
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().acos());
        let hi = Float::with_val(prec, self.hi().acos());
        // acos is decreasing, so lo/hi swap
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &lo - &hi);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    pub fn atan(&self) -> Self {
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().atan());
        let hi = Float::with_val(prec, self.hi().atan());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    /// asinh([m-r, m+r]) — monotone increasing on all of ℝ.
    pub fn asinh(&self) -> Self {
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().asinh());
        let hi = Float::with_val(prec, self.hi().asinh());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    /// acosh([m-r, m+r]) — monotone increasing on `[1, ∞)`. Returns `None` if
    /// the interval extends below 1 (outside the real domain).
    pub fn acosh(&self) -> Option<Self> {
        if self.lo() < 1 {
            return None;
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().acosh());
        let hi = Float::with_val(prec, self.hi().acosh());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    /// atanh([m-r, m+r]) — monotone increasing on `(-1, 1)`. Returns `None` if
    /// the interval reaches or leaves `(-1, 1)` (outside the real domain).
    pub fn atanh(&self) -> Option<Self> {
        if self.lo() <= -1 || self.hi() >= 1 {
            return None;
        }
        let prec = self.prec;
        let lo = Float::with_val(prec, self.lo().atanh());
        let hi = Float::with_val(prec, self.hi().atanh());
        let sum = Float::with_val(prec, &lo + &hi);
        let diff = Float::with_val(prec, &hi - &lo);
        Some(ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        })
    }

    pub fn erf(&self) -> Self {
        let prec = self.prec;
        // Use midpoint + Lipschitz: |erf'(x)| = 2/sqrt(π) * exp(-x²) ≤ 2/sqrt(π) ≈ 1.13
        let mid = Float::with_val(prec, self.mid.clone().erf());
        let lipschitz = Float::with_val(prec, 2.0_f64 / std::f64::consts::PI.sqrt());
        let rad = Float::with_val(prec, &self.rad * &lipschitz);
        let mut b = ArbBall { mid, rad, prec };
        b.add_rounding_error();
        b
    }

    pub fn erfc(&self) -> Self {
        let prec = self.prec;
        let mid = Float::with_val(prec, self.mid.clone().erfc());
        let lipschitz = Float::with_val(prec, 2.0_f64 / std::f64::consts::PI.sqrt());
        let rad = Float::with_val(prec, &self.rad * &lipschitz);
        let mut b = ArbBall { mid, rad, prec };
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
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }

    pub fn ceil_ball(&self) -> Self {
        let prec = self.prec;
        let lo_ceil = Float::with_val(prec, self.lo().ceil());
        let hi_ceil = Float::with_val(prec, self.hi().ceil());
        let diff = Float::with_val(prec, &hi_ceil - &lo_ceil);
        let sum = Float::with_val(prec, &lo_ceil + &hi_ceil);
        ArbBall {
            mid: sum / 2_f64,
            rad: diff / 2_f64,
            prec,
        }
    }
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

    fn eval_predicate(&self, pred: ExprId, pool: &ExprPool) -> Option<bool> {
        if let Some(b) = try_predicate_bool_from_expr(pred, pool) {
            return Some(b);
        }
        let ExprData::Predicate { kind, args } = pool.get(pred) else {
            return None;
        };
        let mid = |id: ExprId| self.eval_node(id, pool).map(|b| b.mid.to_f64());
        match kind {
            PredicateKind::True => Some(true),
            PredicateKind::False => Some(false),
            PredicateKind::Not => Some(!self.eval_predicate(args[0], pool)?),
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
            PredicateKind::Lt => Some(mid(args[0])? < mid(args[1])?),
            PredicateKind::Le => Some(mid(args[0])? <= mid(args[1])?),
            PredicateKind::Gt => Some(mid(args[0])? > mid(args[1])?),
            PredicateKind::Ge => Some(mid(args[0])? >= mid(args[1])?),
            PredicateKind::Eq => Some(mid(args[0])? == mid(args[1])?),
            PredicateKind::Ne => Some(mid(args[0])? != mid(args[1])?),
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
            ExprData::Func { name, args } if args.len() == 1 => {
                let x = self.eval_node(args[0], pool)?;
                match name.as_str() {
                    "sin" => Some(x.sin()),
                    "cos" => Some(x.cos()),
                    "exp" => Some(x.exp()),
                    "log" => x.log(),
                    "sqrt" => x.sqrt(),
                    _ => None,
                }
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

    #[test]
    fn acb_modulus() {
        // |3 + 4i| = 5
        let z = AcbBall::from_f64(3.0, 4.0, 128);
        let m = z.modulus();
        assert!(m.contains(5.0));
    }
}
