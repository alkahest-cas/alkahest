//! The imaginary error function `erfi(x) = −i·erf(i·x)`.
//!
//! ```text
//! erfi(x) = (2/√π) ∫₀ˣ e^{t²} dt = (2/√π) Σₙ x^{2n+1} / (n!·(2n+1)),
//! erfi′(x) = (2/√π)·e^{x²}.
//! ```
//!
//! DLMF 7.2.3 (`erfi` is SymPy's, SciPy's and Mathematica's `Erfi`).  It is
//! real, entire, odd and strictly increasing on ℝ, and it is the closed form of
//! `∫ e^{a x²} dx` for `a > 0` — the half of the Gaussian family that `erf`
//! does not cover.
//!
//! # Numerics
//!
//! Every term of the Maclaurin series has the sign of `x`, so the series has
//! **no cancellation** at any argument: summing it at a modest working
//! precision is accurate for every `x` whose value is representable
//! (`|x| ≲ 26.6`, where `e^{x²}` leaves `f64`).  The number of terms grows like
//! `x²`, which the cap `MAX_TERMS` bounds.
//!
//! The truncation is rigorous: once `n + 1 ≥ 2x²` the ratio of consecutive
//! `aₙ = |x|^{2n+1}/n!` is `x²/(n+1) ≤ ½`, so the discarded tail is at most
//! `2·a_{N+1}`.  That is what the ball kernel's radius is built from.

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::primitive::Primitive;
use rug::float::{Constant, Round};
use rug::Float;

/// Hard cap on series terms (`|x| = 26.6` needs about 800).
const MAX_TERMS: usize = 4096;

/// `(2/√π)·Σ x^{2n+1}/(n!(2n+1))` at `prec` bits, with a rigorous bound on the
/// absolute error.  `None` for a non-finite argument or past the term cap.
fn erfi_float(x: &Float, prec: u32) -> Option<(Float, Float)> {
    if !x.is_finite() {
        return None;
    }
    if x.is_zero() {
        return Some((Float::with_val(prec, 0), Float::with_val(prec, 0)));
    }
    let work = prec + 64;
    let ax = Float::with_val(work, x.clone().abs());
    let x2 = Float::with_val(work, &ax * &ax);
    let x2f = x2.to_f64();
    if !x2f.is_finite() {
        return None;
    }
    let mut a = ax.clone(); // aₙ = |x|^{2n+1}/n!
    let mut sum = Float::with_val(work, &a);
    let mut n_terms = 1usize;
    let mut tail = None;
    for n in 1..MAX_TERMS {
        a *= &x2;
        a /= n as u32;
        sum += Float::with_val(work, &a / (2 * n + 1) as u32);
        n_terms += 1;
        if (n as f64) + 1.0 >= 2.0 * x2f {
            let next = Float::with_val(work, Float::with_val(work, &a * &x2) / (n + 1) as u32);
            let thresh = Float::with_val(work, &sum >> (prec + 8));
            if next < thresh {
                tail = Some(Float::with_val(work, next * 2u32));
                break;
            }
        }
    }
    let tail = tail?;
    let scale = Float::with_val(
        work,
        Float::with_val(work, Constant::Pi).sqrt().recip() * 2u32,
    );
    let mut value = Float::with_val(work, &sum * &scale);
    // Rounding: each of the ~3n operations is off by ≤ ½ ulp of a quantity no
    // larger than the final sum.
    let rounding = Float::with_val(
        work,
        Float::with_val(work, &sum >> (work - 4)) * (3 * n_terms) as u32,
    );
    let err = Float::with_val_round(
        prec,
        Float::with_val(work, Float::with_val(work, tail + rounding) * 2u32),
        Round::Up,
    )
    .0;
    if *x < 0 {
        value = -value;
    }
    Some((Float::with_val(prec, value), err))
}

/// `erfi(x)` in `f64`.  `None` past the overflow of `e^{x²}`.
pub fn erfi(x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    let (v, _) = erfi_float(&Float::with_val(53, x), 80)?;
    let out = v.to_f64();
    out.is_finite().then_some(out)
}

/// `erfi` over a ball.  `erfi` is increasing on ℝ, so the hull of the two
/// endpoint enclosures is an enclosure of the image.
pub fn ball_erfi(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let (a, ea) = erfi_float(&lo, prec)?;
    let (c, ec) = erfi_float(&hi, prec)?;
    let work = prec + 32;
    let l = Float::with_val_round(work, Float::with_val(work, &a - &ea), Round::Down).0;
    let h = Float::with_val_round(work, Float::with_val(work, &c + &ec), Round::Up).0;
    let mid = Float::with_val(prec, Float::with_val(work, &l + &h) / 2u32);
    let r1 = Float::with_val(work, Float::with_val(work, &h - &mid).abs());
    let r2 = Float::with_val(work, Float::with_val(work, &mid - &l).abs());
    let widest = if r1 > r2 { r1 } else { r2 };
    let rad = Float::with_val_round(prec, &widest, Round::Up).0;
    Some(ArbBall { mid, rad, prec })
}

fn two_over_sqrt_pi(pool: &ExprPool) -> ExprId {
    let root = pool.func("sqrt", vec![crate::eval::symbols::pi_symbol(pool)]);
    pool.mul(vec![
        pool.integer(2_i32),
        pool.pow(root, pool.integer(-1_i32)),
    ])
}

fn exp_x2(x: ExprId, pool: &ExprPool) -> ExprId {
    pool.func("exp", vec![pool.pow(x, pool.integer(2_i32))])
}

/// `erfi(x)`, the imaginary error function (DLMF 7.2.3).
/// `erfi′(x) = (2/√π)·e^{x²}`.
pub struct ErfiPrimitive;

impl Primitive for ErfiPrimitive {
    fn name(&self) -> &'static str {
        "erfi"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        format!("erfi({})", pool.display(args[0]))
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        let x = args[0];
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        Some(pool.mul(vec![two_over_sqrt_pi(pool), exp_x2(x, pool), dx]))
    }

    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        let x = args[0];
        Some(vec![pool.mul(vec![
            cotan,
            two_over_sqrt_pi(pool),
            exp_x2(x, pool),
        ])])
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        if args.len() != 1 {
            return None;
        }
        erfi(args[0])
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        if args.len() != 1 {
            return None;
        }
        ball_erfi(&args[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    /// Reference values: `erfi(x) = −i·erf(ix)`, computed with mpmath at 30
    /// digits (`mpmath.erfi`).
    #[test]
    fn matches_reference_values() {
        for (x, want) in [
            (0.1, 0.113_215_174_169_599_8),
            (0.5, 0.614_952_094_696_511),
            (1.0, 1.650_425_758_797_542_9),
            (2.0, 18.564_802_414_575_553),
            (3.0, 1_629.994_622_601_565_7),
        ] {
            let got = erfi(x).unwrap();
            assert!(rel(got, want) < 1e-14, "erfi({x}) = {got}, want {want}");
            assert_eq!(erfi(-x).unwrap(), -got, "odd at {x}");
        }
        assert_eq!(erfi(0.0), Some(0.0));
    }

    /// Far out the series is still free of cancellation; compare against the
    /// asymptotic form `e^{x²}/(x√π)·(1 + 1/(2x²) + 3/(4x⁴) + 15/(8x⁶))`.
    #[test]
    fn large_arguments_follow_the_asymptotic_form() {
        for x in [8.0_f64, 12.0, 20.0] {
            let x2 = x * x;
            let asym = (x2).exp() / (x * std::f64::consts::PI.sqrt())
                * (1.0 + 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2) + 15.0 / (8.0 * x2 * x2 * x2));
            assert!(rel(erfi(x).unwrap(), asym) < 1e-6, "erfi({x})");
        }
        assert_eq!(erfi(40.0), None, "e^1600 is not an f64");
    }

    #[test]
    fn ball_encloses_a_high_precision_value() {
        for x in [-2.5_f64, -0.3, 0.0, 0.7, 1.9] {
            let b = ArbBall::from_midpoint_radius(x, 1e-12, 128);
            let out = ball_erfi(&b).unwrap();
            let (v, e) = erfi_float(&Float::with_val(200, x), 200).unwrap();
            assert!(e < 1e-50, "error bound at {x}");
            assert!(out.contains(v.to_f64()), "erfi ball at {x} misses {v}");
            assert!(out.rad.to_f64() < 1e-9, "erfi ball at {x} is too wide");
        }
    }
}
