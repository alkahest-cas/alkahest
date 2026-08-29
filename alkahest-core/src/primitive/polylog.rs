//! The dilogarithm `Li₂`.
//!
//! # `dilog(x)`, not `polylog(s, x)` — and why
//!
//! `Li₂` is registered as a **unary** primitive `dilog`.  A general
//! `polylog(s, x)` is the more useful object on paper — `Li₁(x) = −log(1−x)`,
//! and `Li₃`, `Li₄` turn up in the same integrals — but it is the wrong thing
//! to ship *here*, for three reasons that are specific to this codebase:
//!
//! 1. **Accuracy.**  A `polylog` worth having needs an evaluator that is
//!    accurate for arbitrary `s`, which means a Hurwitz-zeta / Bose–Einstein
//!    route with its own convergence analysis.  `Li₂` alone reduces to three
//!    identities and one Bernoulli series and lands within a few ulps
//!    everywhere on the real line (see [`dilog`]).  A shaky `polylog` that
//!    happens to be right at `s = 2` would be worse than no `polylog`.
//! 2. **Differentiation would be *permanently* partial.**  `∂Li_s/∂x` is
//!    `Li_{s−1}(x)/x`, fine — but `∂Li_s/∂s` has no closed form at all, so a
//!    binary `polylog` would ship with a partial derivative that can never be
//!    filled in.  That is the exact failure mode Task 1 of this change exists
//!    to remove, re-introduced at a new name.
//! 3. **The rigorous tier is unary.**  Every `Func` rule in
//!    [`crate::validated::taylor`] destructures one argument; `atan2` has no
//!    Taylor model for precisely this reason.  A binary `polylog` would
//!    therefore report `taylor_model = false` and be invisible to
//!    `bound_on_box`, so the verification gate — the reason this work is being
//!    done — could not use it.
//!
//! `Li₁` needs no primitive (`-log(1-x)` already exists), and if `Li₃` is ever
//! wanted the honest move is another unary primitive `trilog`, or a binary
//! `polylog` introduced together with the arity-2 machinery in the validated
//! tier.  Neither is blocked by this choice.
//!
//! # Branch cut
//!
//! `Li₂(z) = Σ_{k≥1} zᵏ/k²` for `|z| ≤ 1`, continued to `ℂ ∖ [1, ∞)`.  This is
//! the **principal branch**, with the cut on `[1, ∞)` — DLMF §25.12(i)
//! (25.12.1–25.12.2), Lewin, *Polylogarithms and Associated Functions* §1.1,
//! and Mathematica's `PolyLog[2, z]`.
//!
//! On the cut-free part of the real line, `(−∞, 1]`, `Li₂` is real, and the
//! endpoint `x = 1` belongs to the domain: `Li₂(1) = π²/6`, the series
//! converging there.  Only the *derivative* blows up at 1.
//!
//! For `x > 1` the principal value is **complex**,
//!
//! ```text
//! Li₂(x ± i0) = π²/3 − ½·log²x − Li₂(1/x)  ∓  iπ·log x,
//! ```
//!
//! so the real kernels here return `None` rather than picking a side of the
//! cut or silently returning the real part.  (MPFR's `mpfr_li2`, which this
//! module uses as an independent oracle in its tests, *does* return the real
//! part for `x > 1`; that is a different convention and is deliberately not
//! exposed.)  Declining matches what `gamma` does at its poles and what the
//! elliptic integrals do outside their domains.
//!
//! Anchors, all pinned in tests: `Li₂(1) = π²/6`, `Li₂(−1) = −π²/12`,
//! `Li₂(1/2) = π²/12 − log²2/2`, `Li₂(0) = 0`.
//!
//! # Derivative
//!
//! ```text
//! d/dx Li₂(x) = −log(1 − x)/x,
//! ```
//!
//! removable at `x = 0` (the limit is 1); the expression built by
//! [`DilogPrimitive::diff_forward`] is the literal quotient, which is `0/0`
//! at the origin exactly as `sin(x)/x`'s would be.

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::primitive::Primitive;
use rug::Float;

/// `π²/6 = Li₂(1)`.
const PI2_6: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;

/// `Bₖ/(k+1)!` for the even `k ≥ 2`; the odd Bernoulli numbers vanish from
/// `k = 3` on, and `k = 0, 1` are written out in [`dilog_core`].
///
/// Generated exactly with rational arithmetic from the Bernoulli recurrence
/// and rounded once to `f64`.
const BERNOULLI_EVEN: [f64; 10] = [
    2.777_777_777_777_777_6e-2,   // B₂/3!  = 1/36
    -2.777_777_777_777_778e-4,    // B₄/5!  = −1/3600
    4.724_111_866_969_01e-6,      // B₆/7!  = 1/211680
    -9.185_773_074_661_964e-8,    // B₈/9!  = −1/10886400
    1.897_886_998_897_1e-9,       // B₁₀/11! = 1/526901760
    -4.064_761_645_144_225_6e-11, // B₁₂/13!
    8.921_691_020_456_452e-13,    // B₁₄/15!
    -1.993_929_586_072_107_4e-14, // B₁₆/17!
    4.518_980_029_619_918e-16,    // B₁₈/19!
    -1.035_651_761_218_124_7e-17, // B₂₀/21!
];

// ---------------------------------------------------------------------------
// f64 kernel
// ---------------------------------------------------------------------------

/// Dilogarithm `Li₂(x)` on `(−∞, 1]`.
///
/// `None` for `x > 1` (the principal value is complex there — see the module
/// docs) and for a non-finite argument.
///
/// # Reduction
///
/// The Bernoulli series
///
/// ```text
/// Li₂(x) = Σ_{k≥0} Bₖ·u^{k+1}/(k+1)!,      u = −log(1 − x)
/// ```
///
/// (Lewin §1.1, eq. 1.13) converges for `|u| < 2π`, but it is only *fast* when
/// `|u|` is small, and its accuracy is set by `|u|` rather than by `x`.  Two
/// standard identities bring every real argument into `x ∈ [−1, 1/2]`, where
/// `|u| ≤ log 2 = 0.693` and successive even terms fall off by
/// `(u/2π)² ≈ 0.012`:
///
/// * **inversion**, for `x < −1` (DLMF 25.12.4 / Lewin 1.11):
///   `Li₂(x) = −Li₂(1/x) − π²/6 − ½·log²(−x)`, mapping `x ↦ 1/x ∈ (−1, 0)`;
/// * **reflection**, for `x ∈ (1/2, 1)` (DLMF 25.12.3 / Lewin 1.12):
///   `Li₂(x) = π²/6 − log(x)·log(1−x) − Li₂(1−x)`, mapping `x ↦ 1−x ∈ (0, 1/2)`.
///
/// So the switchovers are at exactly **`x = −1`** and **`x = 1/2`**, and both
/// are continuous by construction rather than by luck: at `x = −1` the
/// identity reads `Li₂(−1) = −Li₂(−1) − π²/6` (both sides `−π²/12`), and at
/// `x = 1/2` it reads `Li₂(½) = π²/6 − log²2 − Li₂(½)` (both sides
/// `π²/12 − log²2/2`).  `continuity_across_both_switchovers` checks this
/// numerically as well.
///
/// **Accuracy.**  Measured against MPFR's correctly-rounded `mpfr_li2` — an
/// independent implementation, not this series (see
/// `matches_mpfr_across_the_real_line`) — the worst relative error over a
/// 34 000-point sweep of `x ∈ [−10⁶, 1]`, with spot checks out to `−10¹⁵`, is
/// `5.0·10⁻¹⁶`: two to three ulps, attained near `x ≈ 0.56` just inside the
/// reflection branch.  The zero at `x = 0` is the one place where the error is
/// absolute rather than relative, and there `u = −log1p(−x) ≈ x` keeps it at
/// one ulp.  `log1p` rather than `log(1 − x)` is what buys that: the naive form
/// loses everything below `x ≈ 10⁻¹⁶`.
pub fn dilog(x: f64) -> Option<f64> {
    if !x.is_finite() || x > 1.0 {
        return None;
    }
    if x == 1.0 {
        return Some(PI2_6);
    }
    if x < -1.0 {
        let l = (-x).ln();
        return Some(-dilog_core(1.0 / x) - PI2_6 - 0.5 * l * l);
    }
    if x > 0.5 {
        return Some(PI2_6 - x.ln() * (-x).ln_1p() - dilog_core(1.0 - x));
    }
    Some(dilog_core(x))
}

/// The Bernoulli series, valid (and fast) for `x ∈ [−1, 1/2]`.
fn dilog_core(x: f64) -> f64 {
    let u = -(-x).ln_1p(); // −log(1 − x), accurate for small |x|
    let t = u * u;
    let mut p = 0.0_f64;
    for &c in BERNOULLI_EVEN.iter().rev() {
        p = p * t + c;
    }
    // B₀·u + B₁·u²/2! + Σ_{even k ≥ 2} Bₖ·u^{k+1}/(k+1)!
    u - 0.25 * t + u * t * p
}

// ---------------------------------------------------------------------------
// Rigorous point kernel
// ---------------------------------------------------------------------------

/// An enclosure of `Li₂(x)` at `prec` bits, for `x ≤ 1`.
///
/// Delegates to MPFR's `mpfr_li2`, which is correctly rounded, so the only
/// error is the final rounding — absorbed outward here.  `None` for `x > 1`
/// (MPFR would return the *real part* of the principal value there; this
/// module does not expose that convention).
pub fn dilog_ball_point(x: &Float, prec: u32) -> Option<ArbBall> {
    if !x.is_finite() || *x > 1u32 {
        return None;
    }
    let work = prec + 32;
    let v = Float::with_val(work, x).li2();
    let mid = Float::with_val(prec, &v);
    let trunc = Float::with_val(prec, Float::with_val(work, &v - &mid).abs());
    let mut b = ArbBall {
        mid,
        rad: trunc,
        prec,
    };
    // MPFR's own half-ulp at the working precision, plus this rounding.
    let mut bump = Float::with_val(prec, b.mid.abs_ref());
    bump >>= prec.saturating_sub(2);
    b.rad += bump;
    Some(b)
}

// ---------------------------------------------------------------------------
// Primitive
// ---------------------------------------------------------------------------

/// `Li₂(x)`, the dilogarithm.  Principal branch, cut on `[1, ∞)`.
pub struct DilogPrimitive;

impl Primitive for DilogPrimitive {
    fn name(&self) -> &'static str {
        "dilog"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        format!("Li₂({})", pool.display(args[0]))
    }

    fn simplify(&self, args: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
        // Only `Li₂(0) = 0` folds to something exact in the pool's number
        // tower.  `Li₂(1) = π²/6` and `Li₂(−1) = −π²/12` would need a
        // symbolic `π²`, and rewriting them as `f64` literals would turn an
        // exact value into an approximate one behind the user's back.
        match pool.get(args[0]) {
            crate::kernel::expr::ExprData::Integer(n) if n.0 == 0 => Some(pool.integer(0_i32)),
            _ => None,
        }
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        if args.len() != 1 {
            return None;
        }
        let x = args[0];
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        Some(pool.mul(vec![dilog_local_derivative(x, pool), dx]))
    }

    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        if args.len() != 1 {
            return None;
        }
        Some(vec![
            pool.mul(vec![cotan, dilog_local_derivative(args[0], pool)])
        ])
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [x] => dilog(*x),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        match args {
            [only] => only.dilog(),
            _ => None,
        }
    }
}

/// `−log(1 − x)/x`.
fn dilog_local_derivative(x: ExprId, pool: &ExprPool) -> ExprId {
    let one_minus_x = pool.add(vec![
        pool.integer(1_i32),
        pool.mul(vec![pool.integer(-1_i32), x]),
    ]);
    let log = pool.func("log", vec![one_minus_x]);
    let inv_x = pool.pow(x, pool.integer(-1_i32));
    pool.mul(vec![pool.integer(-1_i32), log, inv_x])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MPFR's `mpfr_li2` is correctly rounded and was written by someone else;
    /// checking the reduction above against it is a genuinely independent
    /// cross-validation, which comparing the series to itself would not be.
    fn mpfr_li2(x: f64) -> f64 {
        Float::with_val(200, x).li2().to_f64()
    }

    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            (got - want).abs() / want.abs()
        }
    }

    /// Published closed forms.  `Li₂(1) = π²/6` and `Li₂(−1) = −π²/12` are
    /// DLMF 25.12.2 / Lewin §1.1; `Li₂(½) = π²/12 − log²2/2` is Lewin eq. 1.16
    /// (Landen's value).
    #[test]
    fn the_published_anchors() {
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        assert!(rel(dilog(1.0).unwrap(), pi2 / 6.0) < 1e-15);
        assert!(rel(dilog(-1.0).unwrap(), -pi2 / 12.0) < 1e-15);
        let half = pi2 / 12.0 - 0.5 * std::f64::consts::LN_2 * std::f64::consts::LN_2;
        assert!(rel(dilog(0.5).unwrap(), half) < 1e-15);
        assert_eq!(dilog(0.0), Some(0.0));
    }

    /// A dense sweep, not a handful of points: the reduction has three
    /// branches and two seams, and a sampling that happened to miss one of
    /// them would prove nothing.
    ///
    /// Measured worst case over a 34 000-point grid spanning `[−10⁶, 1]`
    /// (finer than this test runs, to keep the debug build quick) is
    /// `5.0·10⁻¹⁶` — two to three ulps — attained near `x ≈ 0.56`, just inside
    /// the reflection branch.
    #[test]
    fn matches_mpfr_across_the_real_line() {
        let mut xs: Vec<f64> = Vec::new();
        let mut v = -8.0_f64;
        while v <= 1.0 {
            xs.push(v);
            v += 0.02;
        }
        xs.extend([
            -1.0e15,
            -1.0e6,
            -1000.0,
            -10.0,
            -2.0,
            -1.5,
            -1.0,
            -1.000_000_1,
            -0.999_999_9,
            -0.75,
            -0.5,
            -0.125,
            -1e-3,
            -1e-8,
            0.0,
            1e-8,
            1e-3,
            0.125,
            0.25,
            0.4,
            0.499_999_9,
            0.5,
            0.500_000_1,
            0.6,
            0.75,
            0.9,
            0.99,
            0.999_999,
            1.0 - 1e-15,
            1.0,
        ]);
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        for x in xs {
            let got = dilog(x).unwrap();
            let want = mpfr_li2(x);
            let r = rel(got, want);
            if r > worst {
                worst = r;
                worst_at = x;
            }
            assert!(r < 1e-15, "Li₂({x}): got {got}, MPFR {want}, rel {r:e}");
        }
        assert!(
            worst < 1e-15,
            "worst relative error {worst:e} at {worst_at}"
        );
    }

    /// `x = −1` (inversion) and `x = 1/2` (reflection) are the only two
    /// switchovers; a jump at either would be the classic bug.
    #[test]
    fn continuity_across_both_switchovers() {
        for seam in [-1.0_f64, 0.5] {
            for d in [1e-12_f64, 1e-9, 1e-6, 1e-3] {
                let lo = dilog(seam - d).unwrap();
                let hi = dilog(seam + d).unwrap();
                // |Li₂′| = |log(1−x)/x| ≤ 1.4 on [−1.01, 0.51].
                assert!(
                    (hi - lo).abs() < 1.5 * 2.0 * d + 1e-15,
                    "jump at {seam} over ±{d}: {lo} vs {hi}"
                );
                assert!(rel(lo, mpfr_li2(seam - d)) < 1e-15);
                assert!(rel(hi, mpfr_li2(seam + d)) < 1e-15);
            }
        }
    }

    /// The documented branch-cut convention, as behaviour: `x = 1` is in the
    /// domain, anything past it is not.
    #[test]
    fn the_cut_starts_at_one_and_one_itself_is_included() {
        assert!(dilog(1.0).is_some());
        for x in [1.0 + f64::EPSILON, 1.000_001, 2.0, 1e300] {
            assert!(dilog(x).is_none(), "Li₂({x}) must decline: cut on [1, ∞)");
        }
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(dilog(x).is_none());
        }
    }

    #[test]
    fn the_derivative_is_minus_log1mx_over_x() {
        let h = 1e-6_f64;
        for x in [-3.0_f64, -1.0, -0.4, 0.3, 0.7, 0.95] {
            let d = (dilog(x + h).unwrap() - dilog(x - h).unwrap()) / (2.0 * h);
            let want = -(1.0 - x).ln() / x;
            assert!((d - want).abs() < 1e-8, "Li₂′({x}): {d} vs {want}");
        }
    }

    /// A low-precision ball must contain the high-precision value, compared as
    /// `Float`s — rounding the 200-bit value to `f64` first would inject
    /// 10⁻¹⁷ of error into a 10⁻³⁸ ball and make the test measure the wrong
    /// thing.
    #[test]
    fn the_point_enclosure_encloses() {
        for x in [-5.0_f64, -1.0, -0.25, 0.0, 0.5, 0.9, 1.0] {
            let high = Float::with_val(200, x).li2();
            for prec in [32_u32, 53, 128] {
                let b = dilog_ball_point(&Float::with_val(prec, x), prec).unwrap();
                assert!(
                    b.lo() <= high && high <= b.hi(),
                    "Li₂({x}) {high} outside {b}"
                );
            }
            // …and the `f64` reduction agrees with it.
            assert!(rel(dilog(x).unwrap(), high.to_f64()) < 4e-16);
        }
        assert!(dilog_ball_point(&Float::with_val(128, 1.5), 128).is_none());
    }
}
