//! Rigorous **global** bounds via Taylor models and validated numerics.
//!
//! [`crate::ball`] gives rigorous *pointwise* enclosures: evaluate `f` at a
//! ball and the true value is inside the returned ball.  That is not enough
//! for statements quantified over a whole region — "the maximum of `f` on
//! `[a,b]×[c,d]` is at most `M`", "`∫_a^b f dx ∈ [I₁, I₂]`", "`f` has no root
//! in this box".  Plain interval evaluation over a wide box *is* rigorous but
//! suffers the **dependency problem**: `x - x` on `x ∈ [0,1]` evaluates to
//! `[-1,1]` instead of `{0}`, and the looseness compounds multiplicatively
//! through an expression tree.
//!
//! A **Taylor model** fixes this.  On a box `B` with midpoint `c` and radius
//! vector `r`, write `x = c + r ⊙ u` with `u ∈ [-1,1]ⁿ`.  A Taylor model of
//! `f` is a pair `(P, I)` where `P` is a polynomial in `u` with ball
//! coefficients and `I` is an interval, such that
//!
//! ```text
//! ∀ u ∈ [-1,1]ⁿ :  f(c + r ⊙ u) ∈ P(u) + I
//! ```
//!
//! Because the polynomial part tracks *correlations* between subexpressions,
//! cancellation like `x - x` happens symbolically in `P` rather than being
//! lost to interval widening, and the remainder shrinks like `O(h^{p+1})`
//! under subdivision instead of `O(h)`.
//!
//! # Soundness contract
//!
//! Everything in this module returns **outer** enclosures.  Specifically:
//!
//! * Every arithmetic step is performed in [`ArbBall`] ball arithmetic, which
//!   grows the radius by a rounding term after each operation.  Wherever a
//!   raw [`rug::Float`] bound is extracted, it is inflated outward first (see
//!   [`ub`] / [`lb`]).
//! * Truncated polynomial terms are never dropped — they are bounded over
//!   `[-1,1]ⁿ` and folded into the remainder interval.
//! * Elementary functions use a Lagrange remainder whose derivative bound is
//!   taken over an enclosure of the *whole* argument range, not at a point.
//! * If any step cannot be bounded rigorously — an unsupported primitive, a
//!   singularity or branch cut inside the box, a non-finite enclosure — the
//!   operation **refuses** with a structured [`ValidatedError`] rather than
//!   returning a bound that might be wrong.
//!
//! The consequence is that a returned enclosure may be **wide**, but it is
//! never **wrong**.  A wide-but-true bound is a fine answer for a search loop;
//! a tight-but-false one is catastrophic.
//!
//! # Three-valued answers
//!
//! Predicates ([`bounds::verified_no_roots`], [`bounds::verified_sign`])
//! return a [`bounds::Verdict`] with three cases — `True`, `False` and
//! `Undecided`.  `Undecided` means the budget or precision ran out before the
//! question could be settled; it is never collapsed into either of the other
//! two.
//!
//! # Example
//!
//! ```
//! use alkahest_cas::kernel::{Domain, ExprPool};
//! use alkahest_cas::validated::bounds::{bound_on_box, BoundOptions};
//!
//! let pool = ExprPool::new();
//! let x = pool.symbol("x", Domain::Real);
//! // f(x) = x - x  — zero everywhere, but naive interval arithmetic says [-1, 1].
//! let minus_one = pool.integer(-1_i32);
//! let f = pool.add(vec![x, pool.mul(vec![minus_one, x])]);
//!
//! let r = bound_on_box(f, &pool, &[(x, -1.0, 1.0)], &BoundOptions::default()).unwrap();
//! assert!(r.lower() >= -1e-20 && r.upper() <= 1e-20);
//! ```

pub mod bounds;
/// Directed-rounding interval arithmetic over the extended reals — the
/// last-resort range bound for one panel of `bounds::verified_integral`.
mod interval;
pub mod taylor;

use crate::ball::ArbBall;
use crate::errors::AlkahestError;
use rug::Float;
use std::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure modes of the validated-numerics subsystem.
///
/// Every variant is a **refusal**: the requested rigorous statement could not
/// be established, so no bound is returned at all.  This is deliberate — a
/// guessed bound would defeat the purpose of the module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidatedError {
    /// The expression contains a node or function with no rigorous Taylor
    /// model rule (`E-VALIDATED-001`).
    Unsupported {
        /// Human-readable description of the offending construct.
        what: String,
    },
    /// A free symbol was not given an interval in the box (`E-VALIDATED-002`).
    UnboundSymbol {
        /// The symbol name.
        name: String,
    },
    /// The box crosses a singularity, branch cut or domain boundary
    /// (`E-VALIDATED-003`).
    DomainViolation {
        /// Which operation, and why it is undefined here.
        what: String,
    },
    /// An enclosure became infinite or NaN, so nothing can be certified
    /// (`E-VALIDATED-004`).
    NotFinite {
        /// Where the overflow happened.
        what: String,
    },
    /// Malformed request: empty box, inverted interval, bad order, and so on
    /// (`E-VALIDATED-005`).
    InvalidInput {
        /// What is wrong with the request.
        what: String,
    },
}

impl fmt::Display for ValidatedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidatedError::Unsupported { what } => {
                write!(f, "no rigorous Taylor model rule for {what}")
            }
            ValidatedError::UnboundSymbol { name } => {
                write!(f, "symbol `{name}` has no interval in the box")
            }
            ValidatedError::DomainViolation { what } => {
                write!(f, "domain violation on the box: {what}")
            }
            ValidatedError::NotFinite { what } => {
                write!(f, "enclosure is not finite: {what}")
            }
            ValidatedError::InvalidInput { what } => write!(f, "invalid request: {what}"),
        }
    }
}

impl std::error::Error for ValidatedError {}

impl AlkahestError for ValidatedError {
    fn code(&self) -> &'static str {
        match self {
            ValidatedError::Unsupported { .. } => "E-VALIDATED-001",
            ValidatedError::UnboundSymbol { .. } => "E-VALIDATED-002",
            ValidatedError::DomainViolation { .. } => "E-VALIDATED-003",
            ValidatedError::NotFinite { .. } => "E-VALIDATED-004",
            ValidatedError::InvalidInput { .. } => "E-VALIDATED-005",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ValidatedError::Unsupported { .. } => Some(
                "rewrite the expression using +, -, *, /, integer powers, exp, log, sqrt, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, erf or erfc; piecewise and non-smooth nodes cannot be Taylor-modelled",
            ),
            ValidatedError::UnboundSymbol { .. } => {
                Some("give every free symbol an interval in the box argument")
            }
            ValidatedError::DomainViolation { .. } => Some(
                "shrink the box so the argument stays strictly inside the function's domain, or split the box around the singularity and bound each piece separately",
            ),
            ValidatedError::NotFinite { .. } => Some(
                "raise the working precision, lower the Taylor order, or shrink the box — the intermediate enclosure overflowed",
            ),
            ValidatedError::InvalidInput { .. } => {
                Some("check that lo <= hi for every variable and that order >= 1")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interval helpers built on ArbBall
// ---------------------------------------------------------------------------

/// `2^{-(prec-2)}` as an exact `Float` — the outward-rounding safety margin.
///
/// Four ulps of headroom: every helper here performs at most a couple of
/// round-to-nearest operations, each of which can move the result by half an
/// ulp, so four ulps strictly dominates the accumulated error.
fn eps(prec: u32) -> Float {
    let mut e = Float::with_val(prec, 1);
    e >>= prec.saturating_sub(2);
    e
}

/// Grow the radius of `b` outward so that any round-to-nearest performed while
/// constructing it cannot have shrunk the enclosure.
pub fn inflate(b: &ArbBall) -> ArbBall {
    let prec = b.prec;
    if !b.mid.is_finite() || !b.rad.is_finite() {
        return ArbBall::infinity(prec);
    }
    let magnitude = Float::with_val(prec, b.mid.abs_ref()) + b.rad.clone();
    let bump = Float::with_val(prec, magnitude * eps(prec));
    let mut out = b.clone();
    out.rad += bump;
    out
}

/// A rigorous **upper** bound for every value in `b`.
pub fn ub(b: &ArbBall) -> Float {
    inflate(b).hi()
}

/// A rigorous **lower** bound for every value in `b`.
pub fn lb(b: &ArbBall) -> Float {
    inflate(b).lo()
}

/// Build the interval `[lo, hi]` as a ball, rounding outward.
pub fn from_bounds(lo: &Float, hi: &Float, prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, Float::with_val(prec, lo + hi) / 2u32);
    let rad = Float::with_val(prec, Float::with_val(prec, hi - lo) / 2u32).abs();
    inflate(&ArbBall { mid, rad, prec })
}

/// A ball centred at zero with radius `r` — the interval `[-r, r]`.
pub fn symmetric(r: &Float, prec: u32) -> ArbBall {
    inflate(&ArbBall {
        mid: Float::new(prec),
        rad: Float::with_val(prec, r).abs(),
        prec,
    })
}

/// Re-round an arbitrary `Float` into a ball at working precision, outward.
pub fn from_float(v: &Float, prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, v);
    let wide = Float::with_val(prec + 16, v);
    let diff = Float::with_val(prec + 16, wide - &mid).abs();
    inflate(&ArbBall {
        mid,
        rad: Float::with_val(prec, diff),
        prec,
    })
}

/// `max{ |x| : x ∈ b }`, rounded up.
pub fn mag(b: &ArbBall) -> Float {
    let prec = b.prec;
    let m = Float::with_val(prec, b.mid.abs_ref()) + b.rad.clone();
    let bump = Float::with_val(prec, &m * eps(prec));
    Float::with_val(prec, m + bump)
}

/// `min{ |x| : x ∈ b }` (the *mignitude*), rounded down.  Zero when `b`
/// straddles the origin.
pub fn mig(b: &ArbBall) -> Float {
    let prec = b.prec;
    let lo = b.lo();
    let hi = b.hi();
    if lo <= 0 && hi >= 0 {
        return Float::new(prec);
    }
    let a = Float::with_val(prec, lo.abs_ref());
    let c = Float::with_val(prec, hi.abs_ref());
    let m = if a < c { a } else { c };
    let bump = Float::with_val(prec, &m * eps(prec));
    let out = Float::with_val(prec, &m - &bump);
    if out < 0 {
        Float::new(prec)
    } else {
        out
    }
}

/// Smallest ball containing both `a` and `b`.
pub fn hull(a: &ArbBall, b: &ArbBall) -> ArbBall {
    let prec = a.prec.max(b.prec);
    let (alo, ahi, blo, bhi) = (a.lo(), a.hi(), b.lo(), b.hi());
    let lo = if alo < blo { alo } else { blo };
    let hi = if ahi > bhi { ahi } else { bhi };
    from_bounds(&lo, &hi, prec)
}

/// True when `0 ∈ b`.
pub fn contains_zero(b: &ArbBall) -> bool {
    b.lo() <= 0 && b.hi() >= 0
}

/// True when both endpoints of `b` are finite.
pub fn is_finite(b: &ArbBall) -> bool {
    b.mid.is_finite() && b.rad.is_finite()
}

/// Width `hi - lo` of the enclosure, rounded up.
pub fn width(b: &ArbBall) -> Float {
    let prec = b.prec;
    let w = Float::with_val(prec, &b.rad * 2u32);
    let bump = Float::with_val(prec, &w * eps(prec));
    Float::with_val(prec, w + bump)
}

/// π enclosed as a ball.
pub fn pi_ball(prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, rug::float::Constant::Pi);
    inflate(&ArbBall {
        mid,
        rad: Float::new(prec),
        prec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u32 = 128;

    #[test]
    fn inflate_only_grows() {
        let b = ArbBall::from_midpoint_radius(1.0, 0.5, P);
        let i = inflate(&b);
        assert!(i.rad >= b.rad);
        assert!(i.lo() <= b.lo());
        assert!(i.hi() >= b.hi());
    }

    #[test]
    fn from_bounds_encloses_endpoints() {
        let lo = Float::with_val(P, -1.25);
        let hi = Float::with_val(P, 3.5);
        let b = from_bounds(&lo, &hi, P);
        assert!(b.lo() <= lo);
        assert!(b.hi() >= hi);
        assert!(b.contains(0.0));
    }

    #[test]
    fn mag_and_mig() {
        let b = from_bounds(&Float::with_val(P, 2.0), &Float::with_val(P, 5.0), P);
        assert!(mag(&b) >= 5.0);
        assert!(mig(&b) <= 2.0);
        assert!(mig(&b) > 1.9);

        let straddling = from_bounds(&Float::with_val(P, -1.0), &Float::with_val(P, 2.0), P);
        assert_eq!(mig(&straddling), 0.0);
        assert!(mag(&straddling) >= 2.0);
    }

    #[test]
    fn hull_covers_both() {
        let a = from_bounds(&Float::with_val(P, 0.0), &Float::with_val(P, 1.0), P);
        let b = from_bounds(&Float::with_val(P, 3.0), &Float::with_val(P, 4.0), P);
        let h = hull(&a, &b);
        assert!(h.lo() <= 0.0);
        assert!(h.hi() >= 4.0);
    }

    #[test]
    fn error_codes_are_stable() {
        let e = ValidatedError::Unsupported {
            what: "gamma".into(),
        };
        assert_eq!(e.code(), "E-VALIDATED-001");
        assert!(e.remediation().is_some());
        assert_eq!(
            ValidatedError::UnboundSymbol { name: "y".into() }.code(),
            "E-VALIDATED-002"
        );
        assert_eq!(
            ValidatedError::DomainViolation { what: "log".into() }.code(),
            "E-VALIDATED-003"
        );
        assert_eq!(
            ValidatedError::NotFinite { what: "exp".into() }.code(),
            "E-VALIDATED-004"
        );
        assert_eq!(
            ValidatedError::InvalidInput { what: "box".into() }.code(),
            "E-VALIDATED-005"
        );
    }

    #[test]
    fn pi_ball_contains_pi() {
        // Compare against π at *higher* precision, not the f64 constant: the
        // ball encloses π to ~2^-126, and the f64 approximation of π is a
        // whole 1e-17 away from it, so `contains(f64::consts::PI)` would be
        // asking the wrong question.
        let p = pi_ball(P);
        let truth = Float::with_val(P + 64, rug::float::Constant::Pi);
        assert!(p.lo() <= truth && truth <= p.hi());
    }
}
