//! The exponential-integral family: `Ei`, `li`, `Si`, `Ci`, `Shi`, `Chi`.
//!
//! Six primitives whose common structure is that each is the antiderivative of
//! an *elementary* integrand:
//!
//! | primitive | derivative | entire? |
//! |---|---|---|
//! | `Ei(x)`  | `eˣ/x`     | no — logarithmic singularity at `0` |
//! | `li(x)`  | `1/log x`  | no — logarithmic singularity at `1` |
//! | `Si(x)`  | `sin(x)/x` | yes |
//! | `Ci(x)`  | `cos(x)/x` | no — logarithmic singularity at `0` |
//! | `Shi(x)` | `sinh(x)/x`| yes |
//! | `Chi(x)` | `cosh(x)/x`| no — logarithmic singularity at `0` |
//!
//! Because every derivative is elementary, an antiderivative built from this
//! family can be checked by the ordinary `d/dx F = f` verification gate with
//! no new machinery — which is why all six carry a `diff_forward` and a
//! `diff_reverse` rule.  A primitive without a derivative rule cannot appear
//! in a *verified* antiderivative at all.
//!
//! # Conventions, branch cuts and domains
//!
//! Every choice below matches [DLMF §6.2](https://dlmf.nist.gov/6.2)
//! (equivalently Abramowitz & Stegun §5.1–5.2); none of it is invented here.
//!
//! * **`Ei(x)`** — DLMF 6.2.5.  `Ei(x) = ⨍_{-∞}^{x} eᵗ/t dt`, a Cauchy
//!   principal value for `x > 0`.  It is real for every real `x ≠ 0` and
//!   satisfies `Ei(-z) = -E₁(z)` for `z > 0` (DLMF 6.2.1, 6.2.6), which is the
//!   identity the negative branch is computed from.  `Ei(0)` is `-∞` from both
//!   sides; the kernel returns `f64::NEG_INFINITY` rather than refusing.
//!   `Ei` is **not** `E₁`, and it is not `-E₁(-x)` for `x > 0` — the two
//!   differ by `±iπ` off the positive axis, and mixing them up is the classic
//!   way to ship a confidently wrong `∫eˣ/x dx`.
//!
//! * **`li(x)`** — DLMF 6.2.8.  `li(x) = ⨍₀ˣ dt/log t`, again a Cauchy
//!   principal value, taken through the singularity at `t = 1` whenever
//!   `x > 1`.  `li(x) = Ei(log x)` (DLMF 6.2.8) is the identity used, so the
//!   principal value is inherited from `Ei` rather than reimplemented.
//!   Domain: `x ≥ 0`.  `li(0) = 0`, `li(1) = -∞`, and `li(x)` for `x < 0` is
//!   complex — the real kernel **refuses** (`None`) there rather than
//!   returning a real part.
//!
//! * **`Si(x)`** — DLMF 6.2.9.  `Si(x) = ∫₀ˣ (sin t)/t dt`; entire, odd,
//!   `Si(±∞) = ±π/2`.  Defined for every real `x`.
//!
//! * **`Ci(x)`** — DLMF 6.2.11/6.2.13.  `Ci(x) = -∫ₓ^∞ (cos t)/t dt
//!   = γ + log x + ∫₀ˣ (cos t - 1)/t dt` for `x > 0`.  `Ci` has a branch cut
//!   along the negative real axis: `Ci(-x) = Ci(x) ± iπ` (DLMF 6.4.6), so
//!   there is **no** real value on `x < 0`.  The kernel refuses (`None`)
//!   there.  `Ci(0) = -∞`.
//!
//! * **`Shi(x)`** — DLMF 6.2.15.  `Shi(x) = ∫₀ˣ (sinh t)/t dt`; entire, odd,
//!   defined for every real `x`.
//!
//! * **`Chi(x)`** — DLMF 6.2.16.  `Chi(x) = γ + log x + ∫₀ˣ (cosh t - 1)/t dt`
//!   for `x > 0`.  Same branch cut as `Ci`: `Chi(-x) = Chi(x) ± iπ`, so the
//!   kernel refuses (`None`) on `x < 0` and returns `-∞` at `0`.
//!
//! Refusing on the negative reals for `Ci`/`Chi`/`li` is deliberate.  Silently
//! returning the real part of a complex value is the failure mode this file
//! exists to avoid; `None` propagates as an honest "no real value here".
//!
//! # Numerical strategy
//!
//! Every function here is `γ + log|x|` (or nothing) plus a slice of the entire
//! series
//!
//! ```text
//! Σ_{m≥1} σ(m)·xᵐ/(m·m!),   σ(m) ∈ {0, ±1}
//! ```
//!
//! (DLMF 6.6.4, 6.6.5, 6.6.7, 6.6.9, 6.6.10).  The `f64` kernels use that
//! series near the origin and switch to a method with no cancellation further
//! out — a Taylor series alone is badly wrong once `|x|` is more than a few
//! units, because the terms peak at `m ≈ |x|` with magnitude
//! `≈ e^{|x|}/(|x|√(2π|x|))` while the answer stays `O(1)`.  Switchover points
//! and the reasons for them are documented at each kernel.
//!
//! The arbitrary-precision (`ArbBall`) kernels take the opposite trade: they
//! use *only* the series, at a working precision inflated by an explicit
//! cancellation guard, and carry a rigorous absolute error bound.  That makes
//! them an independent implementation of the same functions, which is what the
//! cross-check tests at the bottom of this file exploit.

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::primitive::Primitive;
use crate::special::EULER_GAMMA;
use crate::validated::taylor::TaylorModel;
use crate::validated::{is_finite, mag, symmetric, ValidatedError};
use rug::float::{Constant, Round};
use rug::{Float, Integer, Rational};

type VResult<T> = std::result::Result<T, ValidatedError>;

// ===========================================================================
// Which subseries of Σ xᵐ/(m·m!) a given function uses
// ===========================================================================

/// The sign pattern `σ(m)` selecting one of the five subseries of
/// `Σ_{m≥1} σ(m)·xᵐ/(m·m!)`.
///
/// | kind | `σ(m)` | function |
/// |---|---|---|
/// | `All`      | `1`                        | `Ei(x) − γ − log(abs x)` |
/// | `OddAlt`   | `(−1)^{(m−1)/2}`, `m` odd  | `Si(x)` |
/// | `EvenAlt`  | `(−1)^{m/2}`, `m` even     | `Ci(x) − γ − log x` |
/// | `OddPlus`  | `1`, `m` odd               | `Shi(x)` |
/// | `EvenPlus` | `1`, `m` even              | `Chi(x) − γ − log x` |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeriesKind {
    All,
    OddAlt,
    EvenAlt,
    OddPlus,
    EvenPlus,
}

impl SeriesKind {
    /// `σ(m)`, or `None` when the term is absent from this subseries.
    fn sigma(self, m: usize) -> Option<i32> {
        match self {
            SeriesKind::All => Some(1),
            SeriesKind::OddAlt => (m % 2 == 1).then(|| if ((m - 1) / 2) % 2 == 0 { 1 } else { -1 }),
            SeriesKind::EvenAlt => (m % 2 == 0).then_some(if (m / 2) % 2 == 0 { 1 } else { -1 }),
            SeriesKind::OddPlus => (m % 2 == 1).then_some(1),
            SeriesKind::EvenPlus => (m % 2 == 0).then_some(1),
        }
    }

    /// Does the function add the `γ + log|x|` prefix to the series?
    fn has_log_prefix(self) -> bool {
        matches!(
            self,
            SeriesKind::All | SeriesKind::EvenAlt | SeriesKind::EvenPlus
        )
    }

    /// Bits of extra working precision needed to absorb the cancellation the
    /// subseries suffers at argument magnitude `ax`.
    ///
    /// The largest term is `≈ e^{ax}/(ax·√(2π·ax))`.  A subseries with a
    /// constant sign has no cancellation at all; an alternating one cancels
    /// down to `O(1)` (`Si`, `Ci`) or to `e^{−ax}` (`Ei` on the negative axis,
    /// which is `−E₁`).  `log₂ e ≈ 1.4427`.
    fn guard_bits(self, ax: f64, x_negative: bool) -> u32 {
        let ratio = match self {
            SeriesKind::All if x_negative => 2.0 * ax,
            SeriesKind::All | SeriesKind::OddPlus | SeriesKind::EvenPlus => 0.0,
            SeriesKind::OddAlt | SeriesKind::EvenAlt => ax,
        };
        (ratio * std::f64::consts::LOG2_E).ceil().max(0.0) as u32
    }
}

// ===========================================================================
// f64 kernels
// ===========================================================================

/// Above this the `Ei` power series is replaced by the asymptotic expansion.
///
/// The series `γ + log x + Σ xᵐ/(m·m!)` has *only positive terms* for `x > 0`,
/// so it never cancels: at `x = 45` it needs ≈ 140 terms and loses nothing but
/// accumulated rounding (`≈ √140 · 2⁻⁵³ ≈ 1e-15` relative).  The asymptotic
/// series `(eˣ/x)·Σ k!/xᵏ` (DLMF 6.12.2) is optimally truncated at `k ≈ x`
/// with error `≈ e^{−x}√(2πx)`, which is `≈ 5e-19` at `x = 45` but `2e-8` at
/// `x = 20`.  45 is the smallest round crossover at which *both* branches are
/// good to ~1e-16 relative, which is what makes the seam invisible.
const EI_SERIES_MAX: f64 = 45.0;

/// Below this `E₁` uses its power series, above it the continued fraction.
///
/// The `E₁` series alternates, losing `≈ e^{2z}`; at `z = 1` that is under one
/// decimal digit.  The continued fraction (A&S 5.1.22) converges for every
/// `z > 0` but needs progressively more iterations as `z → 0`, so 1 is the
/// conventional meeting point (it is also Numerical Recipes' `expint` split).
const E1_CF_MIN: f64 = 1.0;

/// Below this `Si`/`Ci` use their power series, above it the complex continued
/// fraction for `E₁(ix)`.
///
/// `Si`/`Ci` alternate and lose `≈ eˣ`: about 2.6 decimal digits at `x = 6`
/// and 8.7 at `x = 20` — the series alone would be down to nine digits well
/// inside the range users care about.  The continued fraction is comfortably
/// convergent by `x = 6`.
const SICI_SERIES_MAX: f64 = 6.0;

/// Below this `Shi`/`Chi` use their power series, above it `Ei`/`E₁`.
///
/// Unlike `Si`/`Ci` the `Shi`/`Chi` series has only positive terms and never
/// cancels, so this crossover is about *cost*, not accuracy: past `x = 6` the
/// identities `Shi = (Ei(x) + E₁(x))/2` and `Chi = (Ei(x) − E₁(x))/2`
/// (immediate from `Ei(±x) = Chi(x) ± Shi(x)`) are cheaper, and `E₁(x)` is
/// exponentially small there so neither combination cancels either.
const SHICHI_SERIES_MAX: f64 = 6.0;

/// Relative tolerance at which the `f64` power series stop.  Below one ulp on
/// purpose: the terms are still shrinking, so one extra term is free accuracy.
const F64_SERIES_EPS: f64 = 1e-18;

/// Relative tolerance at which the `f64` continued fractions stop.  One ulp,
/// not less — a Lentz recurrence that keeps iterating past convergence
/// *accumulates* rounding rather than removing it.
const F64_CF_EPS: f64 = 2.3e-16;

/// `E₁(z) = ∫_z^∞ e^{−t}/t dt` for `z > 0` (DLMF 6.2.1).
///
/// `E₁(0) = +∞`.  Negative `z` is not this function's job — `E₁` has a branch
/// cut on the negative reals, and [`ei`] is the real continuation there.
pub fn e1(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z <= 0.0 {
        return f64::INFINITY;
    }
    if z < E1_CF_MIN {
        // DLMF 6.6.2: E₁(z) = −γ − log z + Σ_{m≥1} (−1)^{m+1} zᵐ/(m·m!).
        let mut term = 1.0_f64;
        let mut sum = 0.0_f64;
        for m in 1..=200 {
            term *= z / m as f64;
            let c = term / m as f64;
            sum += if m % 2 == 1 { c } else { -c };
            if c < F64_SERIES_EPS * (sum.abs() + 1.0) {
                break;
            }
        }
        -EULER_GAMMA - z.ln() + sum
    } else {
        // A&S 5.1.22, evaluated by modified Lentz:
        //   E₁(z) = e^{−z} / (z + 1 − 1²/(z + 3 − 2²/(z + 5 − …)))
        const FPMIN: f64 = 1e-300;
        let mut b = z + 1.0;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=1000 {
            let a = -(i as f64) * (i as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() <= F64_CF_EPS {
                break;
            }
        }
        h * (-z).exp()
    }
}

/// `Ei(x)` for `x > 0` — see [`ei`].
fn ei_positive(x: f64) -> f64 {
    if x <= EI_SERIES_MAX {
        // DLMF 6.6.4: Ei(x) = γ + log x + Σ_{m≥1} xᵐ/(m·m!).
        let mut term = 1.0_f64;
        let mut sum = 0.0_f64;
        for m in 1..=500 {
            term *= x / m as f64;
            let c = term / m as f64;
            sum += c;
            if c < F64_SERIES_EPS * sum {
                break;
            }
        }
        EULER_GAMMA + x.ln() + sum
    } else {
        // DLMF 6.12.2: Ei(x) ~ (eˣ/x)·Σ_{k≥0} k!/xᵏ, truncated at the
        // smallest term — the series is divergent, so taking one term past the
        // minimum makes the answer worse rather than better.
        let mut sum = 1.0_f64;
        let mut term = 1.0_f64;
        let mut prev = f64::INFINITY;
        for k in 1..=200 {
            term *= k as f64 / x;
            if term > prev {
                break;
            }
            prev = term;
            sum += term;
            if term < F64_SERIES_EPS * sum {
                break;
            }
        }
        x.exp() / x * sum
    }
}

/// `Ei(x) = ⨍_{−∞}^x eᵗ/t dt`, the exponential integral, for every real `x`.
///
/// `Ei(0) = −∞` (from both sides).  `None` only for NaN.
pub fn ei(x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    if x == 0.0 {
        return Some(f64::NEG_INFINITY);
    }
    Some(if x < 0.0 {
        // DLMF 6.2.6: Ei(−z) = −E₁(z) for z > 0.
        -e1(-x)
    } else {
        ei_positive(x)
    })
}

/// `li(x) = ⨍₀ˣ dt/log t`, the logarithmic integral (DLMF 6.2.8).
///
/// Cauchy principal value through the singularity at `t = 1` when `x > 1`.
/// `None` for `x < 0`, where `li` is complex.  `li(0) = 0`, `li(1) = −∞`.
pub fn li(x: f64) -> Option<f64> {
    if x.is_nan() || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if x == 1.0 {
        return Some(f64::NEG_INFINITY);
    }
    ei(x.ln())
}

/// `(Si(x), Ci(x))` from the shared power series, for `x > 0`.
fn si_ci_series(x: f64) -> (f64, f64) {
    // DLMF 6.6.5 / 6.6.7, sharing the running term xᵐ/m!:
    //   Si(x) = Σ_{m odd}  (−1)^{(m−1)/2} xᵐ/(m·m!)
    //   Ci(x) = γ + log x + Σ_{m even} (−1)^{m/2} xᵐ/(m·m!)
    let mut term = 1.0_f64;
    let mut si_sum = 0.0_f64;
    let mut ci_sum = 0.0_f64;
    for m in 1..=400 {
        term *= x / m as f64;
        let c = term / m as f64;
        if m % 2 == 1 {
            si_sum += if ((m - 1) / 2) % 2 == 0 { c } else { -c };
        } else {
            ci_sum += if (m / 2) % 2 == 0 { c } else { -c };
        }
        if (m as f64) > 2.0 * x + 2.0 && c < F64_SERIES_EPS * (si_sum.abs() + ci_sum.abs() + 1.0) {
            break;
        }
    }
    (si_sum, EULER_GAMMA + x.ln() + ci_sum)
}

/// `(Si(x), Ci(x))` from the continued fraction for `E₁(ix)`, for `x > 0`.
fn si_ci_cf(x: f64) -> (f64, f64) {
    // Substituting t = iu in E₁(z) = ∫_z^∞ e^{−t}/t dt gives
    //     E₁(ix) = ∫ₓ^∞ e^{−iu}/u du = −Ci(x) + i·(Si(x) − π/2),
    // using Ci(x) = −∫ₓ^∞ (cos u)/u du (DLMF 6.2.11) and
    // Si(x) − π/2 = −∫ₓ^∞ (sin u)/u du (DLMF 6.2.10).  E₁ itself comes from
    // the same A&S 5.1.22 continued fraction as `e1`, run in complex
    // arithmetic; this is the classical Numerical Recipes `cisi` route.
    const FPMIN: f64 = 1e-300;
    #[inline]
    fn cdiv(ar: f64, ai: f64, br: f64, bi: f64) -> (f64, f64) {
        let den = br * br + bi * bi;
        ((ar * br + ai * bi) / den, (ai * br - ar * bi) / den)
    }
    let (mut br, bi) = (1.0_f64, x);
    let (mut cr, mut cim) = (1.0 / FPMIN, 0.0_f64);
    let (mut dr, mut dim) = cdiv(1.0, 0.0, br, bi);
    let (mut hr, mut him) = (dr, dim);
    for i in 2..=2000 {
        let a = -(((i - 1) * (i - 1)) as f64);
        br += 2.0;
        let (nd_r, nd_i) = cdiv(1.0, 0.0, a * dr + br, a * dim + bi);
        dr = nd_r;
        dim = nd_i;
        let (qr, qi) = cdiv(a, 0.0, cr, cim);
        cr = br + qr;
        cim = bi + qi;
        let (delr, deli) = (cr * dr - cim * dim, cr * dim + cim * dr);
        let (nh_r, nh_i) = (hr * delr - him * deli, hr * deli + him * delr);
        hr = nh_r;
        him = nh_i;
        if (delr - 1.0).abs() + deli.abs() < F64_CF_EPS {
            break;
        }
    }
    // h ← e^{−ix}·h = (cos x − i sin x)·h
    let (cx, sx) = (x.cos(), x.sin());
    let (re, im) = (hr * cx + him * sx, him * cx - hr * sx);
    (std::f64::consts::FRAC_PI_2 + im, -re)
}

/// `Si(x) = ∫₀ˣ (sin t)/t dt` (DLMF 6.2.9).  Entire and odd, so every real
/// `x` — including `±∞`, where the value is `±π/2` — has a value.
pub fn si(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return std::f64::consts::FRAC_PI_2 * x.signum();
    }
    let a = x.abs();
    let s = if a <= SICI_SERIES_MAX {
        si_ci_series(a).0
    } else {
        si_ci_cf(a).0
    };
    if x < 0.0 {
        -s
    } else {
        s
    }
}

/// `Ci(x) = −∫ₓ^∞ (cos t)/t dt` (DLMF 6.2.11).
///
/// `None` for `x < 0`: `Ci(−x) = Ci(x) ± iπ` is not real.  `Ci(0) = −∞`.
pub fn ci(x: f64) -> Option<f64> {
    if x.is_nan() || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(f64::NEG_INFINITY);
    }
    if x.is_infinite() {
        return Some(0.0);
    }
    Some(if x <= SICI_SERIES_MAX {
        si_ci_series(x).1
    } else {
        si_ci_cf(x).1
    })
}

/// `(Shi(x), Chi(x))` from the shared power series, for `x > 0`.
fn shi_chi_series(x: f64) -> (f64, f64) {
    // DLMF 6.6.9 / 6.6.10 — the sign-free analogues of `si_ci_series`.
    let mut term = 1.0_f64;
    let mut shi_sum = 0.0_f64;
    let mut chi_sum = 0.0_f64;
    for m in 1..=400 {
        term *= x / m as f64;
        let c = term / m as f64;
        if m % 2 == 1 {
            shi_sum += c;
        } else {
            chi_sum += c;
        }
        if (m as f64) > x + 2.0 && c < F64_SERIES_EPS * (shi_sum + chi_sum + 1.0) {
            break;
        }
    }
    (shi_sum, EULER_GAMMA + x.ln() + chi_sum)
}

/// `Shi(x) = ∫₀ˣ (sinh t)/t dt` (DLMF 6.2.15).  Entire and odd.
pub fn shi(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x.is_nan() {
        return f64::NAN;
    }
    let a = x.abs();
    let s = if a <= SHICHI_SERIES_MAX {
        shi_chi_series(a).0
    } else {
        // Ei(±x) = Chi(x) ± Shi(x)  ⇒  Shi = (Ei(x) + E₁(x))/2.
        0.5 * (ei_positive(a) + e1(a))
    };
    if x < 0.0 {
        -s
    } else {
        s
    }
}

/// `Chi(x) = γ + log x + ∫₀ˣ (cosh t − 1)/t dt` (DLMF 6.2.16).
///
/// `None` for `x < 0`: `Chi(−x) = Chi(x) ± iπ` is not real.  `Chi(0) = −∞`.
pub fn chi(x: f64) -> Option<f64> {
    if x.is_nan() || x < 0.0 {
        return None;
    }
    if x == 0.0 {
        return Some(f64::NEG_INFINITY);
    }
    Some(if x <= SHICHI_SERIES_MAX {
        shi_chi_series(x).1
    } else {
        0.5 * (ei_positive(x) - e1(x))
    })
}

// ===========================================================================
// Arbitrary-precision kernels
// ===========================================================================

/// Hard stop on the series length.  `|x| ≳ 1000` needs more than this, and at
/// that point the caller is better served by a refusal than by a multi-second
/// evaluation.
const MAX_SERIES_TERMS: usize = 4096;

/// Hard stop on the cancellation guard.  2048 bits covers `E₁` out to
/// `|x| ≈ 700` (where it is already subnormal in `f64`) and `Si`/`Ci` out to
/// `|x| ≈ 1400`.
const MAX_GUARD_BITS: u32 = 2048;

/// `γ` as a ball, correct at `prec` bits.
fn euler_gamma_ball(prec: u32) -> ArbBall {
    let work = prec + 32;
    let g = Float::with_val(work, Constant::Euler);
    ball_around(&g, work, prec)
}

/// A ball at `prec` bits containing `v`, where `v` was itself computed at
/// `work` bits and may be off by up to four ulps there.
fn ball_around(v: &Float, work: u32, prec: u32) -> ArbBall {
    let mid = Float::with_val(prec, v);
    let trunc = Float::with_val(work, Float::with_val(work, v - &mid).abs());
    let slack = Float::with_val(work, Float::with_val(work, v).abs()) >> work.saturating_sub(2);
    let rad = Float::with_val_round(prec, Float::with_val(work, trunc + slack), Round::Up).0;
    ArbBall { mid, rad, prec }
}

/// The smallest ball at `prec` bits containing `[lo, hi]`.
fn ball_from_interval(lo: &Float, hi: &Float, prec: u32) -> ArbBall {
    let work = prec + 32;
    let mid = Float::with_val(prec, Float::with_val(work, lo + hi) / 2u32);
    let r1 = Float::with_val(work, Float::with_val(work, hi - &mid).abs());
    let r2 = Float::with_val(work, Float::with_val(work, &mid - lo).abs());
    let widest = if r1 > r2 { r1 } else { r2 };
    let rad = Float::with_val_round(prec, &widest, Round::Up).0;
    ArbBall { mid, rad, prec }
}

/// Evaluate one member of the family at `x` to `prec` bits, returning the
/// value together with a **rigorous absolute error bound**.
///
/// Unlike the `f64` kernels this uses nothing but the power series — it buys
/// its accuracy with working precision ([`SeriesKind::guard_bits`]) instead of
/// switching algorithm.  That makes it an implementation of the same six
/// functions that shares no code path with the `f64` side beyond the choice of
/// `σ(m)`, which is what the cross-check tests rely on.
///
/// Returns `None` when the argument is outside the kind's real domain
/// (`x ≤ 0` for the `γ + log x` kinds), when the guard precision or the term
/// count would exceed their caps, or when the argument is not finite.
fn eval_float(kind: SeriesKind, x: &Float, prec: u32) -> Option<(Float, Float)> {
    if !x.is_finite() {
        return None;
    }
    if x.is_zero() {
        // Si(0) = Shi(0) = 0 exactly; Ei/Ci/Chi are −∞ there.
        return (!kind.has_log_prefix())
            .then(|| (Float::with_val(prec, 0), Float::with_val(prec, 0)));
    }
    let negative = *x < 0;
    if negative
        && !matches!(
            kind,
            SeriesKind::All | SeriesKind::OddAlt | SeriesKind::OddPlus
        )
    {
        // Ci and Chi are complex on the negative reals — see the module docs.
        return None;
    }
    let ax = Float::with_val(53, x.clone().abs()).to_f64();
    let guard = kind.guard_bits(ax, negative);
    if guard > MAX_GUARD_BITS || !ax.is_finite() {
        return None;
    }
    let work = prec + 96 + guard;

    let xw = Float::with_val(work, x);
    let axw = Float::with_val(work, xw.clone().abs());
    let mut term = Float::with_val(work, 1u32); // xᵐ/m!
    let mut aterm = Float::with_val(work, 1u32); // |x|ᵐ/m!
    let mut sum = Float::with_val(work, 0u32);
    let mut abs_sum = Float::with_val(work, 0u32);
    let mut n_terms = 0usize;
    let mut converged = false;
    for m in 1..=MAX_SERIES_TERMS {
        term *= &xw;
        term /= m as u32;
        aterm *= &axw;
        aterm /= m as u32;
        let ac = Float::with_val(work, &aterm / m as u32);
        if let Some(s) = kind.sigma(m) {
            let c = Float::with_val(work, &term / m as u32);
            if s > 0 {
                sum += &c;
            } else {
                sum -= &c;
            }
            abs_sum += &ac;
        }
        n_terms = m;
        // Past the peak at m ≈ |x| the terms fall off faster than 2⁻ᵐ, so a
        // term below `abs_sum·2^{-work}` bounds the whole remaining tail by
        // twice that — which is inside the rounding error accounted below.
        if (m as f64) > 2.0 * ax + 4.0 {
            let thresh = abs_sum.clone() >> work;
            if ac <= thresh {
                converged = true;
                break;
            }
        }
    }
    if !converged {
        return None;
    }

    let mut val = sum.clone();
    // `err_scale` collects every quantity whose rounding can contaminate the
    // answer, in absolute terms.
    let mut err_scale = abs_sum;
    if kind.has_log_prefix() {
        let g = Float::with_val(work, Constant::Euler);
        let l = Float::with_val(work, axw.ln());
        err_scale += Float::with_val(work, l.clone().abs());
        err_scale += 1u32;
        val += g;
        val += l;
    }
    err_scale += Float::with_val(work, val.clone().abs());
    if !val.is_finite() || !err_scale.is_finite() {
        return None;
    }
    // Four roundings per loop iteration plus a fixed dozen for the prologue
    // and epilogue, each at most one ulp of `err_scale`.
    let mut err = Float::with_val(work, &err_scale * ((4 * n_terms + 16) as u32));
    err >>= work;

    let val_p = Float::with_val(prec, &val);
    let trunc = Float::with_val(work, Float::with_val(work, &val - &val_p).abs());
    let err_p = Float::with_val_round(prec, Float::with_val(work, err + trunc), Round::Up).0;
    Some((val_p, err_p))
}

/// `li(x) = Ei(log x)` at `prec` bits, with a rigorous absolute error bound.
fn li_float(x: &Float, prec: u32) -> Option<(Float, Float)> {
    if !x.is_finite() || *x < 0 {
        return None;
    }
    if x.is_zero() {
        return Some((Float::with_val(prec, 0), Float::with_val(prec, 0)));
    }
    let work = prec + 96;
    let l = Float::with_val(work, x.clone().ln());
    let (v, e) = eval_float(SeriesKind::All, &l, prec)?;
    // `l` is itself off by up to one ulp, i.e. |δl| ≤ |l|·2^{-work}, and
    // |Ei′(l)| = e^l/|l| = x/|l|, so the induced error is at most x·2^{-work}.
    let extra = Float::with_val(work, x.clone().abs()) >> work;
    let err = Float::with_val_round(prec, Float::with_val(work, e + extra), Round::Up).0;
    Some((v, err))
}

/// `Ei` over a ball.  `None` if the ball touches the singularity at `0`.
///
/// `Ei′(x) = eˣ/x` is positive on `(0, ∞)` and negative on `(−∞, 0)`, so `Ei`
/// is monotone on each side of the pole and an endpoint hull is exact.
pub fn ball_ei(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) || (lo <= 0 && hi >= 0) {
        return None;
    }
    let (a, ea) = eval_float(SeriesKind::All, &lo, prec)?;
    let (c, ec) = eval_float(SeriesKind::All, &hi, prec)?;
    let work = prec + 32;
    let (low, high) = if lo > 0 {
        (
            Float::with_val(work, &a - &ea),
            Float::with_val(work, &c + &ec),
        )
    } else {
        (
            Float::with_val(work, &c - &ec),
            Float::with_val(work, &a + &ea),
        )
    };
    Some(ball_from_interval(&low, &high, prec))
}

/// `li` over a ball.  `None` if the ball touches `x = 1` or reaches `x < 0`.
///
/// `li′(x) = 1/log x` is negative on `(0, 1)` and positive on `(1, ∞)`, so an
/// endpoint hull is exact away from the singularity.
pub fn ball_li(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) || lo < 0 || (lo <= 1 && hi >= 1) {
        return None;
    }
    let (a, ea) = li_float(&lo, prec)?;
    let (c, ec) = li_float(&hi, prec)?;
    let work = prec + 32;
    let (low, high) = if lo > 1 {
        (
            Float::with_val(work, &a - &ea),
            Float::with_val(work, &c + &ec),
        )
    } else {
        (
            Float::with_val(work, &c - &ec),
            Float::with_val(work, &a + &ea),
        )
    };
    Some(ball_from_interval(&low, &high, prec))
}

/// `Si` over a ball.
///
/// `Si` oscillates, so an endpoint hull is *not* an enclosure of its range —
/// the same trap `bessel_jn` documents.  `|Si′(x)| = |sin x / x| ≤ 1` for
/// every real `x`, so the mean value theorem gives midpoint + radius instead.
pub fn ball_si(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (v, e) = eval_float(SeriesKind::OddAlt, &b.mid, prec)?;
    let work = prec + 32;
    let rad = Float::with_val_round(
        prec,
        Float::with_val(work, Float::with_val(work, &e + &b.rad)),
        Round::Up,
    )
    .0;
    Some(ArbBall { mid: v, rad, prec })
}

/// `Ci` over a ball.  `None` unless the ball lies strictly inside `(0, ∞)` —
/// `Ci` is complex on the negative reals.
///
/// `Ci` oscillates, so again midpoint + Lipschitz rather than a hull:
/// `|Ci′(x)| = |cos x / x| ≤ 1/lo` on `[lo, hi]` with `lo > 0`.
pub fn ball_ci(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let lo = b.lo();
    if !lo.is_finite() || lo <= 0 || !b.hi().is_finite() {
        return None;
    }
    let (v, e) = eval_float(SeriesKind::EvenAlt, &b.mid, prec)?;
    let work = prec + 32;
    let lip = Float::with_val(work, Float::with_val(work, 1u32) / &lo);
    let rad = Float::with_val_round(
        prec,
        Float::with_val(work, &e + Float::with_val(work, &lip * &b.rad)),
        Round::Up,
    )
    .0;
    Some(ArbBall { mid: v, rad, prec })
}

/// `Shi` over a ball.  `Shi′(x) = sinh(x)/x > 0` for **every** real `x`
/// (including `x = 0`, where it is `1`), so `Shi` is monotone increasing on
/// all of ℝ and an endpoint hull is exact.
pub fn ball_shi(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) {
        return None;
    }
    let (a, ea) = eval_float(SeriesKind::OddPlus, &lo, prec)?;
    let (c, ec) = eval_float(SeriesKind::OddPlus, &hi, prec)?;
    let work = prec + 32;
    Some(ball_from_interval(
        &Float::with_val(work, &a - &ea),
        &Float::with_val(work, &c + &ec),
        prec,
    ))
}

/// `Chi` over a ball.  `None` unless the ball lies strictly inside `(0, ∞)`.
/// `Chi′(x) = cosh(x)/x > 0` there, so an endpoint hull is exact.
pub fn ball_chi(b: &ArbBall) -> Option<ArbBall> {
    let prec = b.prec;
    let (lo, hi) = (b.lo(), b.hi());
    if !(lo.is_finite() && hi.is_finite()) || lo <= 0 {
        return None;
    }
    let (a, ea) = eval_float(SeriesKind::EvenPlus, &lo, prec)?;
    let (c, ec) = eval_float(SeriesKind::EvenPlus, &hi, prec)?;
    let work = prec + 32;
    Some(ball_from_interval(
        &Float::with_val(work, &a - &ea),
        &Float::with_val(work, &c + &ec),
        prec,
    ))
}

// ===========================================================================
// Rigorous Taylor-model rules
// ===========================================================================

/// Hard cap on the degree of the truncated series a Taylor-model rule will
/// evaluate.  Every step is a `TaylorModel::mul`, so the cost is linear in
/// this and quadratic in the model's term count.
///
/// The series is expanded about the **origin**, not about the box centre, so
/// the degree needed grows with `|x|` rather than with the box *width*: the
/// terms `Xᵐ/(m·m!)` do not fall below `2^{-prec}` until `m ≳ e·X`.  256
/// therefore covers argument enclosures out to `|x| ≈ 90`; past that the rule
/// refuses with `E-VALIDATED-004`.  Subdividing does **not** help — a narrow
/// box at `x = 200` has the same `|x|` — so a caller who needs that range
/// wants a rule that re-expands about the box centre, which this is not.
const MAX_TAYLOR_TERMS: usize = 256;

/// Coefficients `c₁ … c_M` of `Σ σ(m)·xᵐ/(m·m!)` together with a rigorous
/// bound on the discarded tail `Σ_{m>M} |c_m|·Xᵐ`, where `X = x_mag`.
///
/// `M` is chosen so that `M + 1 ≥ 2X + 2`.  Past that point
/// `u_{m+1}/u_m = X·m/(m+1)² ≤ X/(m+1) ≤ ½` with `u_m = Xᵐ/(m·m!)`, so the
/// tail is majorised by the geometric series `2·u_{M+1}` — no property of the
/// individual functions is used, only `|σ(m)| ≤ 1`.
fn series_coeffs(kind: SeriesKind, x_mag: &Float, prec: u32) -> VResult<(Vec<ArbBall>, Float)> {
    let work = prec + 32;
    let xf = Float::with_val(53, x_mag).to_f64();
    if !xf.is_finite() {
        return Err(ValidatedError::NotFinite {
            what: "exponential-integral argument enclosure".into(),
        });
    }
    // Scale the target at the size the answer can plausibly reach: these
    // functions are `O(e^X)` at worst (`Ei`, `Chi`, `Shi`).
    let mut target = Float::with_val(work, xf).exp();
    if target < 1 {
        target = Float::with_val(work, 1u32);
    }
    target >>= prec + 4;

    let xw = Float::with_val(work, x_mag);
    let mut coeffs: Vec<ArbBall> = Vec::new();
    let mut fact = Integer::from(1);
    let mut xpow = Float::with_val(work, 1u32);
    let zero = ArbBall::from_f64(0.0, prec);
    for m in 1..=MAX_TAYLOR_TERMS + 1 {
        fact *= m as u32;
        xpow *= &xw;
        let den = Integer::from(m as u32) * fact.clone();
        let u = Float::with_val(work, &xpow / Float::with_val(work, &den));
        if !u.is_finite() {
            return Err(ValidatedError::NotFinite {
                what: "exponential-integral series term".into(),
            });
        }
        if m >= 2 && (m as f64) >= 2.0 * xf + 2.0 && u <= target {
            // `2·u_m` dominates the geometric tail; the extra 2⁻¹⁶ relative
            // slack covers the round-to-nearest in `u` itself, which is
            // `≤ 2^{-(prec+30)}` relative and therefore not close to using it.
            let mut tail = Float::with_val(work, u * 2u32);
            tail += Float::with_val(work, &tail) >> 16u32;
            return Ok((coeffs, Float::with_val_round(prec, tail, Round::Up).0));
        }
        if m > MAX_TAYLOR_TERMS {
            break;
        }
        coeffs.push(match kind.sigma(m) {
            Some(s) => ArbBall::from_rational(&Rational::from((Integer::from(s), den)), prec),
            None => zero.clone(),
        });
    }
    Err(ValidatedError::NotFinite {
        what: format!(
            "exponential-integral series over an argument enclosure of \
             magnitude {xf:.3e}: the origin-centred expansion would need more \
             than {MAX_TAYLOR_TERMS} terms, and subdividing the box does not \
             reduce |x|"
        ),
    })
}

/// `Σ_{m≥1} σ(m)·xᵐ/(m·m!)` as a Taylor model.
///
/// The truncated polynomial is evaluated in the model algebra itself, so the
/// rigour of the result rests only on `TaylorModel::mul`/`add`/`shift` (which
/// already absorb their own truncation into the remainder) plus the tail bound
/// from [`series_coeffs`].  There is no per-function derivative estimate to
/// get wrong.
fn series_model(x: &TaylorModel, kind: SeriesKind) -> VResult<TaylorModel> {
    let prec = x.prec();
    let range = x.range();
    if !is_finite(&range) {
        return Err(ValidatedError::NotFinite {
            what: "exponential-integral argument".into(),
        });
    }
    let (coeffs, tail) = series_coeffs(kind, &mag(&range), prec)?;
    let (nvars, order) = (x.nvars(), x.order());
    let last = coeffs
        .last()
        .cloned()
        .unwrap_or_else(|| ArbBall::from_f64(0.0, prec));
    let mut acc = TaylorModel::constant(last, nvars, order, prec);
    for c in coeffs.iter().rev().skip(1) {
        acc = acc.mul(x).shift(c);
    }
    let poly = acc.mul(x);
    Ok(poly.add(&TaylorModel::constant(
        symmetric(&tail, prec),
        nvars,
        order,
        prec,
    )))
}

/// `log|x|` as a Taylor model, refusing when the enclosure straddles `0`.
fn log_abs_model(x: &TaylorModel, what: &str) -> VResult<TaylorModel> {
    let r = x.range();
    if r.lo() > 0 {
        x.log()
    } else if r.hi() < 0 {
        x.neg().log()
    } else {
        Err(ValidatedError::DomainViolation {
            what: format!("{what} has a logarithmic singularity the box straddles"),
        })
    }
}

/// Rigorous Taylor model for `Ei`.  Refuses when the box contains `0`.
pub fn taylor_ei(x: &TaylorModel) -> VResult<TaylorModel> {
    let g = euler_gamma_ball(x.prec());
    let l = log_abs_model(x, "Ei")?;
    Ok(l.shift(&g).add(&series_model(x, SeriesKind::All)?))
}

/// Rigorous Taylor model for `li(x) = Ei(log x)`.  Refuses when the box
/// reaches `x ≤ 0` or contains the singularity at `x = 1`.
pub fn taylor_li(x: &TaylorModel) -> VResult<TaylorModel> {
    let r = x.range();
    if r.lo() <= 0 {
        return Err(ValidatedError::DomainViolation {
            what: "li(x) is complex for x < 0".into(),
        });
    }
    if r.lo() <= 1 && r.hi() >= 1 {
        return Err(ValidatedError::DomainViolation {
            what: "li has a logarithmic singularity at x = 1 that the box straddles".into(),
        });
    }
    taylor_ei(&x.log()?)
}

/// Rigorous Taylor model for `Si`.  Entire — no domain guard.
pub fn taylor_si(x: &TaylorModel) -> VResult<TaylorModel> {
    series_model(x, SeriesKind::OddAlt)
}

/// Rigorous Taylor model for `Shi`.  Entire — no domain guard.
pub fn taylor_shi(x: &TaylorModel) -> VResult<TaylorModel> {
    series_model(x, SeriesKind::OddPlus)
}

/// Rigorous Taylor model for `Ci`.  Refuses unless the box is strictly
/// positive: `Ci(−x) = Ci(x) ± iπ` is not real.
pub fn taylor_ci(x: &TaylorModel) -> VResult<TaylorModel> {
    let g = euler_gamma_ball(x.prec());
    let l = positive_log_model(x, "Ci")?;
    Ok(l.shift(&g).add(&series_model(x, SeriesKind::EvenAlt)?))
}

/// Rigorous Taylor model for `Chi`.  Refuses unless the box is strictly
/// positive: `Chi(−x) = Chi(x) ± iπ` is not real.
pub fn taylor_chi(x: &TaylorModel) -> VResult<TaylorModel> {
    let g = euler_gamma_ball(x.prec());
    let l = positive_log_model(x, "Chi")?;
    Ok(l.shift(&g).add(&series_model(x, SeriesKind::EvenPlus)?))
}

/// `log x` as a Taylor model, refusing on anything that is not strictly
/// positive — the branch-cut guard shared by `Ci` and `Chi`.
fn positive_log_model(x: &TaylorModel, what: &str) -> VResult<TaylorModel> {
    if x.range().lo() <= 0 {
        return Err(ValidatedError::DomainViolation {
            what: format!("{what}(x) is complex for x ≤ 0"),
        });
    }
    x.log()
}

// ===========================================================================
// Primitive bundles
// ===========================================================================

/// `d/dx wrapped(x) = quotient_num(x)/x · dx` — the shape shared by five of
/// the six (`li` is the odd one out).
fn quotient_diff(args: &[ExprId], wrt: ExprId, pool: &ExprPool, numerator: &str) -> Option<ExprId> {
    let x = args[0];
    let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
    let num = pool.func(numerator, vec![x]);
    let recip = pool.pow(x, pool.integer(-1_i32));
    Some(pool.mul(vec![num, recip, dx]))
}

/// Adjoint of [`quotient_diff`].
fn quotient_adjoint(
    args: &[ExprId],
    cotan: ExprId,
    pool: &ExprPool,
    numerator: &str,
) -> Option<Vec<ExprId>> {
    let x = args[0];
    let num = pool.func(numerator, vec![x]);
    let recip = pool.pow(x, pool.integer(-1_i32));
    Some(vec![pool.mul(vec![cotan, num, recip])])
}

macro_rules! quotient_primitive {
    (
        $struct_name:ident, $name:literal, $numerator:literal,
        $f64_kernel:expr, $ball_kernel:expr, $doc:literal
    ) => {
        #[doc = $doc]
        pub struct $struct_name;

        impl Primitive for $struct_name {
            fn name(&self) -> &'static str {
                $name
            }

            fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
                format!("{}({})", $name, pool.display(args[0]))
            }

            fn diff_forward(
                &self,
                args: &[ExprId],
                wrt: ExprId,
                pool: &ExprPool,
            ) -> Option<ExprId> {
                quotient_diff(args, wrt, pool, $numerator)
            }

            fn diff_reverse(
                &self,
                args: &[ExprId],
                cotan: ExprId,
                pool: &ExprPool,
            ) -> Option<Vec<ExprId>> {
                quotient_adjoint(args, cotan, pool, $numerator)
            }

            fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
                let f: fn(f64) -> Option<f64> = $f64_kernel;
                if args.len() != 1 {
                    return None;
                }
                f(args[0])
            }

            fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
                let f: fn(&ArbBall) -> Option<ArbBall> = $ball_kernel;
                if args.len() != 1 {
                    return None;
                }
                f(&args[0])
            }

            // No `lean_theorem`: the certificate emitter records these under
            // the generic `diff_primitive_registry` rule, which
            // `diff_rule_to_tactic` never maps to a tactic — the same reason
            // `tan`, `gamma` and the hyperbolic family withhold theirs.
        }
    };
}

quotient_primitive!(
    EiPrimitive,
    "Ei",
    "exp",
    ei,
    ball_ei,
    "`Ei(x)`, the exponential integral (DLMF 6.2.5). `Ei′(x) = eˣ/x`."
);

quotient_primitive!(
    SiPrimitive,
    "Si",
    "sin",
    |x| Some(si(x)),
    |b| ball_si(b),
    "`Si(x) = ∫₀ˣ (sin t)/t dt`, the sine integral (DLMF 6.2.9). \
     `Si′(x) = sin(x)/x`."
);

quotient_primitive!(
    CiPrimitive,
    "Ci",
    "cos",
    ci,
    ball_ci,
    "`Ci(x) = −∫ₓ^∞ (cos t)/t dt`, the cosine integral (DLMF 6.2.11). \
     `Ci′(x) = cos(x)/x`.  Complex for `x < 0`, where the kernels refuse."
);

quotient_primitive!(
    ShiPrimitive,
    "Shi",
    "sinh",
    |x| Some(shi(x)),
    |b| ball_shi(b),
    "`Shi(x) = ∫₀ˣ (sinh t)/t dt`, the hyperbolic sine integral \
     (DLMF 6.2.15). `Shi′(x) = sinh(x)/x`."
);

quotient_primitive!(
    ChiPrimitive,
    "Chi",
    "cosh",
    chi,
    ball_chi,
    "`Chi(x) = γ + log x + ∫₀ˣ (cosh t − 1)/t dt`, the hyperbolic cosine \
     integral (DLMF 6.2.16). `Chi′(x) = cosh(x)/x`.  Complex for `x < 0`, \
     where the kernels refuse."
);

/// `li(x) = ⨍₀ˣ dt/log t`, the logarithmic integral (DLMF 6.2.8).
/// `li′(x) = 1/log x`.
pub struct LiPrimitive;

impl Primitive for LiPrimitive {
    fn name(&self) -> &'static str {
        "li"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        format!("li({})", pool.display(args[0]))
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        // d/dx li(x) = 1/log(x) · dx
        let x = args[0];
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        let recip_log = pool.pow(pool.func("log", vec![x]), pool.integer(-1_i32));
        Some(pool.mul(vec![recip_log, dx]))
    }

    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        let x = args[0];
        let recip_log = pool.pow(pool.func("log", vec![x]), pool.integer(-1_i32));
        Some(vec![pool.mul(vec![cotan, recip_log])])
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        if args.len() != 1 {
            return None;
        }
        li(args[0])
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        if args.len() != 1 {
            return None;
        }
        ball_li(&args[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::primitive::{Capabilities, PrimitiveRegistry};

    /// Reference precision for the independent arbitrary-precision path.
    const REF_PREC: u32 = 200;

    /// The arbitrary-precision series, as a plain `f64` reference value.
    ///
    /// This is *not* the same algorithm as the `f64` kernels: it never uses a
    /// continued fraction, an asymptotic expansion, or the
    /// `Shi/Chi ↔ Ei/E₁` identities, and it buys accuracy with working
    /// precision instead.  Checking a series against itself would prove
    /// nothing; checking the `f64` continued fraction against a 200-bit series
    /// is a real cross-validation.
    fn reference(kind: SeriesKind, x: f64) -> f64 {
        eval_float(kind, &Float::with_val(REF_PREC, x), REF_PREC)
            .map(|(v, _)| v.to_f64())
            .unwrap_or(f64::NAN)
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// Error scaled by `max(1, |f|)` — the right yardstick near an isolated
    /// zero, where *no* implementation keeps relative accuracy.
    fn scaled(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1.0)
    }

    // ── Published reference values ──────────────────────────────────────────

    /// Values taken from published sources, not from this file's own series.
    ///
    /// * `Ei(1)`, `Ci(1)`, `li(2)`: DLMF §6.2 / A&S Table 5.1–5.2 (the same
    ///   digits Wolfram's `ExpIntegralEi[1]`, `CosIntegral[1]`,
    ///   `LogIntegral[2]` print).
    /// * `E₁(1) = -Ei(-1) = 0.219383934395520274…`: A&S Table 5.1.
    /// * `Si(π) = 1.851937051982…`: the Wilbraham–Gibbs constant, OEIS
    ///   A036792.
    /// * `Shi`/`Chi` at 0.5, 1, 2, 5: mymathtables.com's
    ///   "Hyperbolic Sine and Cosine Integral Table", 10 significant digits.
    /// * Zeros of `Ci` at 0.616505, 3.38418, 6.42705: MathWorld,
    ///   "Cosine Integral".
    #[test]
    fn published_reference_values() {
        assert!((ei(1.0).unwrap() - 1.895_117_816_355_936_8).abs() < 1e-15);
        assert!((ei(-1.0).unwrap() + 0.219_383_934_395_520_27).abs() < 1e-15);
        assert!((li(2.0).unwrap() - 1.045_163_780_117_493).abs() < 1e-15);
        assert!((ci(1.0).unwrap() - 0.337_403_922_900_968_1).abs() < 1e-15);
        assert!((si(std::f64::consts::PI) - 1.851_937_051_982).abs() < 1e-12);

        // Ten-digit published table (Shi, Chi).
        for &(x, s, c) in &[
            (0.5_f64, 0.506_996_749_8_f64, -0.052_776_844_96_f64),
            (1.0, 1.057_250_875, 0.837_866_941),
            (2.0, 2.501_567_433, 2.452_666_923),
            (5.0, 20.093_211_83, 20.092_063_53),
        ] {
            assert!(rel(shi(x), s) < 5e-10, "Shi({x})");
            assert!(rel(chi(x).unwrap(), c) < 5e-9, "Chi({x})");
        }

        // Ci's first three zeros, to the six digits MathWorld publishes.
        // Ci′ ≈ cos(x)/x is O(1) there, so a six-digit root pins Ci to ~1e-6.
        for &z in &[0.616_505_f64, 3.384_18, 6.427_05] {
            assert!(ci(z).unwrap().abs() < 1e-5, "Ci near published zero {z}");
        }

        // Si(∞) = π/2 and Ci(∞) = 0 (DLMF 6.2.9, 6.2.11).
        assert_eq!(si(f64::INFINITY), std::f64::consts::FRAC_PI_2);
        assert_eq!(si(f64::NEG_INFINITY), -std::f64::consts::FRAC_PI_2);
        assert!((si(1.0e6) - std::f64::consts::FRAC_PI_2).abs() < 2e-6);
        assert!(ci(1.0e6).unwrap().abs() < 2e-6);
    }

    /// `Ei(-x) = -E₁(x)` and `Chi ± Shi = Ei(±x)` are the two identities the
    /// large-argument kernels are built on; if one of them were mis-signed the
    /// numbers would still look plausible.
    #[test]
    fn defining_identities_hold() {
        for &x in &[0.25_f64, 0.75, 1.0, 2.5, 6.0, 6.5, 12.0, 30.0] {
            assert!(rel(ei(-x).unwrap(), -e1(x)) < 1e-14, "Ei(-{x}) = -E1({x})");
            let (s, c) = (shi(x), chi(x).unwrap());
            assert!(rel(c + s, ei(x).unwrap()) < 1e-13, "Chi+Shi = Ei at {x}");
            // `Chi − Shi = Ei(−x)` cancels catastrophically once `x` is
            // more than a few units (both sides are `≈ eˣ/2x`, the difference
            // is `≈ e^{−x}/2x`), so the honest statement is an *absolute*
            // agreement at the scale of the terms being subtracted, not a
            // relative one at the scale of the answer.
            assert!(
                ((c - s) - ei(-x).unwrap()).abs() <= 1e-13 * (c.abs() + s.abs()),
                "Chi-Shi = Ei(-x) at {x}"
            );
        }
        // Odd symmetry of Si and Shi.
        for &x in &[0.5_f64, 3.0, 6.5, 40.0] {
            assert_eq!(si(-x), -si(x));
            assert_eq!(shi(-x), -shi(x));
        }
    }

    /// The `f64` kernels against the independent 200-bit series, everywhere.
    #[test]
    fn f64_kernels_match_the_high_precision_series() {
        let sweep =
            |lo: f64, hi: f64, n: usize| (0..=n).map(move |k| lo + (hi - lo) * k as f64 / n as f64);

        for x in sweep(0.01, 60.0, 600) {
            assert!(
                rel(ei(x).unwrap(), reference(SeriesKind::All, x)) < 1e-13,
                "Ei({x})"
            );
            assert!(
                scaled(si(x), reference(SeriesKind::OddAlt, x)) < 1e-13,
                "Si({x})"
            );
            assert!(
                scaled(ci(x).unwrap(), reference(SeriesKind::EvenAlt, x)) < 1e-13,
                "Ci({x})"
            );
            assert!(
                rel(shi(x), reference(SeriesKind::OddPlus, x)) < 1e-13,
                "Shi({x})"
            );
            assert!(
                rel(chi(x).unwrap(), reference(SeriesKind::EvenPlus, x)) < 1e-13,
                "Chi({x})"
            );
        }
        for x in sweep(-60.0, -0.01, 600) {
            assert!(
                rel(ei(x).unwrap(), reference(SeriesKind::All, x)) < 1e-13,
                "Ei({x})"
            );
            assert!(
                scaled(si(x), reference(SeriesKind::OddAlt, x)) < 1e-13,
                "Si({x})"
            );
            assert!(
                rel(shi(x), reference(SeriesKind::OddPlus, x)) < 1e-13,
                "Shi({x})"
            );
        }
        // `li` past both the singularity and the Ramanujan–Soldner zero.
        for x in sweep(0.01, 0.99, 200).chain(sweep(1.01, 50.0, 400)) {
            let want = li_float(&Float::with_val(REF_PREC, x), REF_PREC)
                .unwrap()
                .0
                .to_f64();
            assert!(scaled(li(x).unwrap(), want) < 1e-13, "li({x})");
        }
        // Far out, where only the asymptotic / continued-fraction branches run.
        for x in sweep(60.0, 300.0, 200) {
            assert!(
                rel(ei(x).unwrap(), reference(SeriesKind::All, x)) < 1e-13,
                "Ei({x})"
            );
            assert!(
                scaled(si(x), reference(SeriesKind::OddAlt, x)) < 1e-13,
                "Si({x})"
            );
            assert!(
                scaled(ci(x).unwrap(), reference(SeriesKind::EvenAlt, x)) < 1e-13,
                "Ci({x})"
            );
        }
    }

    /// A seam between two algorithms is where a wrong constant hides, because
    /// each branch looks fine in isolation.  Evaluate *both* branches at the
    /// crossover and require them to agree — a genuine discontinuity test, not
    /// a finite-difference one (`f(c ± h)` differs by `f′(c)·h` even when the
    /// kernel is perfectly continuous).
    #[test]
    fn branches_agree_at_every_switchover() {
        // Si / Ci at SICI_SERIES_MAX.
        for &x in &[
            SICI_SERIES_MAX,
            SICI_SERIES_MAX - 0.5,
            SICI_SERIES_MAX + 0.5,
        ] {
            let (s_ser, c_ser) = si_ci_series(x);
            let (s_cf, c_cf) = si_ci_cf(x);
            assert!(
                rel(s_ser, s_cf) < 1e-13,
                "Si branches at {x}: {s_ser} vs {s_cf}"
            );
            assert!(
                scaled(c_ser, c_cf) < 1e-13,
                "Ci branches at {x}: {c_ser} vs {c_cf}"
            );
        }
        // Shi / Chi at SHICHI_SERIES_MAX.
        for &x in &[
            SHICHI_SERIES_MAX,
            SHICHI_SERIES_MAX - 0.5,
            SHICHI_SERIES_MAX + 0.5,
        ] {
            let (s_ser, c_ser) = shi_chi_series(x);
            let (s_id, c_id) = (
                0.5 * (ei_positive(x) + e1(x)),
                0.5 * (ei_positive(x) - e1(x)),
            );
            assert!(rel(s_ser, s_id) < 1e-13, "Shi branches at {x}");
            assert!(rel(c_ser, c_id) < 1e-13, "Chi branches at {x}");
        }
        // Ei's series/asymptotic seam: both branches, evaluated at the
        // crossover, against the 200-bit reference.
        for &x in &[EI_SERIES_MAX - 1.0, EI_SERIES_MAX, EI_SERIES_MAX + 1.0] {
            let want = reference(SeriesKind::All, x);
            assert!(rel(ei_positive(x), want) < 1e-13, "Ei kernel at {x}");
        }
        // E₁'s series/continued-fraction seam at z = 1.
        for &z in &[E1_CF_MIN - 0.05, E1_CF_MIN, E1_CF_MIN + 0.05] {
            let want = -reference(SeriesKind::All, -z);
            assert!(rel(e1(z), want) < 1e-13, "E1 kernel at {z}");
        }
        // …and no visible step in the exported functions across each seam.
        for (name, f, c) in [
            (
                "Si",
                Box::new(si) as Box<dyn Fn(f64) -> f64>,
                SICI_SERIES_MAX,
            ),
            ("Ci", Box::new(|v| ci(v).unwrap()), SICI_SERIES_MAX),
            ("Shi", Box::new(shi), SHICHI_SERIES_MAX),
            ("Chi", Box::new(|v| chi(v).unwrap()), SHICHI_SERIES_MAX),
            ("Ei", Box::new(|v| ei(v).unwrap()), EI_SERIES_MAX),
            ("Ei", Box::new(|v| ei(v).unwrap()), -E1_CF_MIN),
        ] {
            let h = 1e-9;
            let (lo, mid, hi) = (f(c - h), f(c), f(c + h));
            // A true discontinuity shows as a second difference far larger
            // than the smooth `f″·h²` a continuous function produces.
            let curvature = (hi - 2.0 * mid + lo).abs() / mid.abs().max(1.0);
            assert!(
                curvature < 1e-12,
                "{name} steps at the seam {c}: {curvature:e}"
            );
        }
    }

    // ── Domain and singularity behaviour ────────────────────────────────────

    #[test]
    fn singular_points_and_branch_cuts() {
        // Logarithmic singularities.
        assert_eq!(ei(0.0), Some(f64::NEG_INFINITY));
        assert_eq!(ci(0.0), Some(f64::NEG_INFINITY));
        assert_eq!(chi(0.0), Some(f64::NEG_INFINITY));
        assert_eq!(li(1.0), Some(f64::NEG_INFINITY));
        // …approached, not just asserted at the point.
        assert!(ei(1e-8).unwrap() < -17.0 && ei(-1e-8).unwrap() < -17.0);
        assert!(ci(1e-8).unwrap() < -17.0);
        assert!(li(1.0 - 1e-8).unwrap() < -17.0 && li(1.0 + 1e-8).unwrap() < -17.0);

        // Finite, defined values.
        assert_eq!(li(0.0), Some(0.0));
        assert_eq!(si(0.0), 0.0);
        assert_eq!(shi(0.0), 0.0);

        // Complex on the negative reals ⇒ refuse, never a real part.
        for &x in &[-1e-12_f64, -0.5, -1.0, -10.0] {
            assert_eq!(ci(x), None, "Ci({x}) must refuse");
            assert_eq!(chi(x), None, "Chi({x}) must refuse");
            assert_eq!(li(x), None, "li({x}) must refuse");
        }
        // Ei is real on both sides of its pole, and Si/Shi are entire.
        assert!(ei(-0.5).unwrap().is_finite());
        assert!(si(-40.0).is_finite() && shi(-40.0).is_finite());

        assert_eq!(ei(f64::NAN), None);
        assert_eq!(li(f64::NAN), None);
        assert_eq!(ci(f64::NAN), None);
        assert_eq!(chi(f64::NAN), None);
        assert!(si(f64::NAN).is_nan());
    }

    // ── Derivative rules ────────────────────────────────────────────────────

    /// Symbolic check: `diff` must produce exactly the elementary derivative,
    /// structurally.  This is the property the integrator's verification gate
    /// consumes.
    #[test]
    fn derivatives_are_the_documented_elementary_functions() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let recip = pool.pow(x, pool.integer(-1_i32));
        for (name, numerator) in [
            ("Ei", "exp"),
            ("Si", "sin"),
            ("Ci", "cos"),
            ("Shi", "sinh"),
            ("Chi", "cosh"),
        ] {
            let d = crate::diff::diff(pool.func(name, vec![x]), x, &pool)
                .expect("diff")
                .value;
            let want = pool.mul(vec![pool.func(numerator, vec![x]), recip]);
            assert_eq!(
                pool.display(d).to_string(),
                pool.display(want).to_string(),
                "d/dx {name}(x) should be {numerator}(x)/x"
            );
        }
        let d = crate::diff::diff(pool.func("li", vec![x]), x, &pool)
            .expect("diff")
            .value;
        let want = pool.pow(pool.func("log", vec![x]), pool.integer(-1_i32));
        assert_eq!(
            pool.display(d).to_string(),
            pool.display(want).to_string(),
            "d/dx li(x) = 1/log x"
        );
    }

    /// Numerical check of the same thing, through the chain rule, on a
    /// composed argument — the case a structural comparison would miss.
    #[test]
    fn derivatives_match_finite_differences() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let arg = pool.add(vec![
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        let reg = PrimitiveRegistry::default_registry();
        for (name, f) in [
            (
                "Ei",
                Box::new(|v: f64| ei(v).unwrap()) as Box<dyn Fn(f64) -> f64>,
            ),
            ("Si", Box::new(si)),
            ("Ci", Box::new(|v| ci(v).unwrap())),
            ("Shi", Box::new(shi)),
            ("Chi", Box::new(|v| chi(v).unwrap())),
            ("li", Box::new(|v| li(v).unwrap())),
        ] {
            let d = crate::diff::diff(pool.func(name, vec![arg]), x, &pool)
                .expect("diff")
                .value;
            for &x0 in &[0.3_f64, 0.9, 2.0] {
                let h = 1e-5;
                let (u, up, um) = (2.0 * x0 + 1.0, 2.0 * (x0 + h) + 1.0, 2.0 * (x0 - h) + 1.0);
                let fd = (f(up) - f(um)) / (2.0 * h);
                let mut env = crate::ball::IntervalEval::new(160);
                env.bind(x, ArbBall::from_f64(x0, 160));
                let got = env.eval(d, &pool).expect("eval derivative").mid_f64();
                assert!(
                    (got - fd).abs() < 1e-6 * fd.abs().max(1.0),
                    "d/dx {name}(2x+1) at {x0}: {got} vs finite difference {fd} (u = {u})"
                );
                // …and the registry's own numeric slot agrees with the kernel.
                assert!(rel(reg.numeric_f64(name, &[u]).unwrap(), f(u)) < 1e-15);
            }
        }
    }

    /// The round trip the integrator will depend on: build `F = Si(x)`,
    /// differentiate, and confirm the result *evaluates* to `sin(x)/x`.  If
    /// this ever fails, an emitted antiderivative would be rejected by the
    /// verification gate even though it is correct.
    #[test]
    fn verification_gate_round_trip() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (name, integrand) in [
            (
                "Ei",
                Box::new(|v: f64| v.exp() / v) as Box<dyn Fn(f64) -> f64>,
            ),
            ("Si", Box::new(|v| v.sin() / v)),
            ("Ci", Box::new(|v| v.cos() / v)),
            ("Shi", Box::new(|v| v.sinh() / v)),
            ("Chi", Box::new(|v| v.cosh() / v)),
            ("li", Box::new(|v| 1.0 / v.ln())),
        ] {
            let big_f = pool.func(name, vec![x]);
            let d = crate::diff::diff(big_f, x, &pool).expect("diff").value;
            for &x0 in &[0.4_f64, 1.7, 3.25, 9.5] {
                let mut env = crate::ball::IntervalEval::new(160);
                env.bind(x, ArbBall::from_f64(x0, 160));
                let got = env.eval(d, &pool).expect("eval").mid_f64();
                let want = integrand(x0);
                assert!(
                    rel(got, want) < 1e-13,
                    "d/dx {name}(x) at {x0}: {got} vs {want}"
                );
            }
        }
    }

    // ── Ball kernels ────────────────────────────────────────────────────────

    #[test]
    fn ball_kernels_enclose_and_agree_with_f64() {
        let prec = 160u32;
        for &x in &[0.25_f64, 1.0, 2.0, 5.9, 6.0, 6.1, 20.0, 44.9, 45.1, 120.0] {
            let b = ArbBall::from_f64(x, prec);
            assert!(rel(ei(x).unwrap(), ball_ei(&b).unwrap().mid_f64()) < 1e-13);
            assert!(scaled(si(x), ball_si(&b).unwrap().mid_f64()) < 1e-13);
            assert!(scaled(ci(x).unwrap(), ball_ci(&b).unwrap().mid_f64()) < 1e-13);
            assert!(rel(shi(x), ball_shi(&b).unwrap().mid_f64()) < 1e-13);
            assert!(rel(chi(x).unwrap(), ball_chi(&b).unwrap().mid_f64()) < 1e-13);
        }
        for &x in &[-0.25_f64, -1.0, -6.5, -30.0] {
            let b = ArbBall::from_f64(x, prec);
            assert!(rel(ei(x).unwrap(), ball_ei(&b).unwrap().mid_f64()) < 1e-13);
            assert!(scaled(si(x), ball_si(&b).unwrap().mid_f64()) < 1e-13);
            assert!(rel(shi(x), ball_shi(&b).unwrap().mid_f64()) < 1e-13);
            assert_eq!(ball_ci(&b), None);
            assert_eq!(ball_chi(&b), None);
            assert_eq!(ball_li(&b), None);
        }

        // A wide ball must *enclose* every sample inside it, not merely have
        // the right midpoint — the failure `bessel_jn` documents for hulls.
        let wide = ArbBall::from_midpoint_radius(4.0, 3.0, prec);
        let si_b = ball_si(&wide).unwrap();
        for k in 0..=400 {
            let v = 1.0 + 6.0 * k as f64 / 400.0;
            assert!(
                si_b.contains(si(v)),
                "Si ball over [1,7] misses Si({v}) = {}",
                si(v)
            );
        }

        // Singularities are refused, not approximated.
        assert_eq!(
            ball_ei(&ArbBall::from_midpoint_radius(0.0, 1.0, prec)),
            None
        );
        assert_eq!(
            ball_li(&ArbBall::from_midpoint_radius(1.0, 0.5, prec)),
            None
        );
        assert_eq!(
            ball_ci(&ArbBall::from_midpoint_radius(0.5, 1.0, prec)),
            None
        );
    }

    // ── Registry / capability wiring ────────────────────────────────────────

    #[test]
    fn every_primitive_registers_a_full_bundle() {
        let reg = PrimitiveRegistry::default_registry();
        for name in ["Ei", "li", "Si", "Ci", "Shi", "Chi"] {
            assert!(reg.is_registered(name), "{name} not registered");
            let caps = reg.capabilities(name);
            for (flag, what) in [
                (Capabilities::NUMERIC_F64, "numeric_f64"),
                (Capabilities::NUMERIC_BALL, "numeric_ball"),
                (Capabilities::DIFF_FORWARD, "diff_forward"),
                (Capabilities::DIFF_REVERSE, "diff_reverse"),
                (Capabilities::TAYLOR_MODEL, "taylor_model"),
            ] {
                assert!(caps.contains(flag), "{name} is missing {what}");
            }
        }
        // Arity is part of the contract: none of these is binary.
        assert_eq!(reg.numeric_f64("Ei", &[1.0, 2.0]), None);
        assert_eq!(reg.numeric_f64("Si", &[]), None);
    }

    // ── Taylor-model rules ──────────────────────────────────────────────────

    #[test]
    fn taylor_models_enclose_the_true_range() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions {
            order: 4,
            prec: 96,
            tol: 1e-10,
            max_subdivisions: 32,
        };
        for &(name, lo, hi) in &[
            ("Ei", 0.25_f64, 0.5_f64),
            ("Ei", 1.0, 3.0),
            ("Ei", -3.0, -1.0),
            ("li", 2.0, 4.0),
            ("li", 0.25, 0.5),
            ("Si", -3.0, 3.0),
            ("Si", 8.0, 12.0),
            ("Ci", 0.5, 2.0),
            ("Ci", 8.0, 12.0),
            ("Shi", -2.0, 2.0),
            ("Chi", 1.0, 3.0),
        ] {
            let e = pool.func(name, vec![x]);
            let r = bound_on_box(e, &pool, &[(x, lo, hi)], &opts)
                .unwrap_or_else(|err| panic!("{name} on [{lo},{hi}]: {err}"));
            let f = |v: f64| match name {
                "Ei" => ei(v).unwrap(),
                "li" => li(v).unwrap(),
                "Si" => si(v),
                "Ci" => ci(v).unwrap(),
                "Shi" => shi(v),
                "Chi" => chi(v).unwrap(),
                _ => unreachable!(),
            };
            for k in 0..=400 {
                let v = lo + (hi - lo) * k as f64 / 400.0;
                let y = f(v);
                assert!(
                    r.lower() <= y + 1e-9 && r.upper() >= y - 1e-9,
                    "{name} enclosure [{}, {}] misses {name}({v}) = {y}",
                    r.lower(),
                    r.upper()
                );
            }
        }
    }

    #[test]
    fn taylor_models_refuse_boxes_that_straddle_a_singularity() {
        use crate::validated::bounds::{bound_on_box, BoundOptions};
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let opts = BoundOptions::default();
        for &(name, lo, hi) in &[
            ("Ei", -1.0_f64, 1.0_f64),
            ("li", 0.5, 2.0),
            ("li", -1.0, 2.0),
            ("Ci", -1.0, 1.0),
            ("Ci", -2.0, -1.0),
            ("Chi", -1.0, 1.0),
            ("Chi", -2.0, -1.0),
        ] {
            let e = pool.func(name, vec![x]);
            assert!(
                bound_on_box(e, &pool, &[(x, lo, hi)], &opts).is_err(),
                "{name} on [{lo},{hi}] must refuse — it is singular or complex there"
            );
        }
    }

    /// The tail bound is the only part of the Taylor rule that is not carried
    /// by `TaylorModel`'s own arithmetic, so pin it directly: truncating the
    /// series at `M` and adding the claimed tail must still bracket the true
    /// value at the edge of the box.
    #[test]
    fn series_tail_bound_is_not_optimistic() {
        for &x in &[0.5_f64, 2.0, 5.0, 12.0] {
            for kind in [
                SeriesKind::All,
                SeriesKind::OddAlt,
                SeriesKind::EvenAlt,
                SeriesKind::OddPlus,
                SeriesKind::EvenPlus,
            ] {
                let xm = Float::with_val(96, x);
                let (coeffs, tail) = series_coeffs(kind, &xm, 96).expect("coeffs");
                // Evaluate the truncated polynomial in plain f64 and compare
                // with the converged series: the gap must fit inside `tail`.
                let mut partial = 0.0_f64;
                let mut pw = 1.0_f64;
                for c in &coeffs {
                    pw *= x;
                    partial += c.mid_f64() * pw;
                }
                let full = reference(kind, x)
                    - if kind.has_log_prefix() {
                        EULER_GAMMA + x.ln()
                    } else {
                        0.0
                    };
                let gap = (full - partial).abs();
                let bound = tail.to_f64();
                assert!(
                    gap <= bound * 1.000_001 + 1e-12 * full.abs().max(1.0),
                    "{kind:?} at x={x}: truncation gap {gap:e} exceeds claimed tail {bound:e}"
                );
            }
        }
    }

    // ── Display ─────────────────────────────────────────────────────────────

    #[test]
    fn display_is_stable_in_both_renderers() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for (name, latex) in [
            ("Ei", r"\operatorname{Ei}"),
            ("li", r"\operatorname{li}"),
            ("Si", r"\operatorname{Si}"),
            ("Ci", r"\operatorname{Ci}"),
            ("Shi", r"\operatorname{Shi}"),
            ("Chi", r"\operatorname{Chi}"),
        ] {
            let e = pool.func(name, vec![x]);
            let tex = crate::kernel::display::render_latex(e, &pool);
            assert!(tex.contains(latex), "latex for {name}: {tex}");
            let uni = crate::kernel::display::render_unicode(e, &pool);
            assert_eq!(uni, format!("{name}(x)"), "unicode for {name}");
            // `Chi` is a function name here, not the Greek letter Χ.
            assert!(!uni.contains('\u{3a7}') && !tex.contains(r"\Chi"));
            assert_eq!(pool.display(e).to_string(), format!("{name}(x)"));
        }
    }
}
