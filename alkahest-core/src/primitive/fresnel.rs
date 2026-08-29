//! Fresnel integrals `S(x)` and `C(x)`.
//!
//! # Convention — normalised (π/2), and why it is stated first
//!
//! Alkahest uses the **normalised** definitions
//!
//! ```text
//! S(x) = ∫₀ˣ sin(π t²/2) dt,        C(x) = ∫₀ˣ cos(π t²/2) dt,
//! ```
//!
//! which are DLMF §7.2(iii) (7.2.7–7.2.8), Abramowitz & Stegun §7.3
//! (7.3.1–7.3.2), SymPy's `fresnels`/`fresnelc`, SciPy's
//! `scipy.special.fresnel` and Mathematica's `FresnelS`/`FresnelC`.  They
//! satisfy `S(∞) = C(∞) = 1/2`.
//!
//! The competing *unnormalised* convention
//! `S₁(x) = ∫₀ˣ sin(t²) dt`, `C₁(x) = ∫₀ˣ cos(t²) dt` (Gradshteyn–Ryzhik, and
//! the "Fresnel integral" of many optics texts) is **not** what these compute.
//! The two differ by an argument scaling *and* an amplitude,
//!
//! ```text
//! S₁(x) = √(π/2)·S(x·√(2/π)),      C₁(x) = √(π/2)·C(x·√(2/π)),
//! ```
//!
//! with `S₁(∞) = C₁(∞) = √(π/8) ≈ 0.6267`, not `1/2`.  Mixing them gives an
//! answer that is wrong by a factor of `√(π/2)` *and* evaluated at the wrong
//! point — a silent error, which is why the convention is pinned here, in the
//! Python docstrings, and in the test `half_at_infinity_pins_the_convention`
//! rather than left implicit.
//!
//! # Derivatives
//!
//! ```text
//! S′(x) = sin(π x²/2),      C′(x) = cos(π x²/2).
//! ```
//!
//! # Numerics
//!
//! Two regimes, switching at `|x| = 6` ([`ASYMPTOTIC_FROM`]):
//!
//! * **`|x| < 6`** — the Maclaurin series (DLMF 7.6.4, 7.6.6), summed in MPFR
//!   at a working precision raised by the series' own cancellation, which is
//!   `(π/2)x²·log₂e ≈ 2.27·x²` bits (57 at `x = 5`, 82 at `x = 6`).  Summed in
//!   plain `f64` this series loses ~10 digits at `x = 4` and ~15 at `x = 5`;
//!   getting that wrong is the classic Fresnel bug and is what the extra
//!   precision buys.
//! * **`|x| ≥ 6`** — the asymptotic expansion in the auxiliary functions
//!   `f`, `g` (DLMF 7.12.1–7.12.3), truncated at its smallest term.  DLMF
//!   §7.12(ii) states that for `ph z = 0` the remainders are bounded in
//!   magnitude by the first neglected terms, which is what makes the
//!   truncation *rigorous* rather than merely plausible.
//!
//! The switchover sits at `6` because that is where both regimes are
//! comfortably below `f64` resolution.  The asymptotic series' smallest term is
//! `≈ √2·e^{−πx²/2}`: `3·10⁻²⁵` at `x = 6` but only `1.5·10⁻¹⁴` at `x = 4.5`,
//! so switching earlier would quietly cap the accuracy — and the power series
//! only needs 82 extra bits at `x = 6`, which is cheap.  Continuity across the
//! seam is pinned by `the_two_regimes_agree_across_the_switchover`.
//!
//! Both functions are odd, so negative arguments are handled by
//! `S(−x) = −S(x)`, `C(−x) = −C(x)` and never enter the kernels.

use crate::ball::ArbBall;
use crate::kernel::{ExprId, ExprPool};
use crate::primitive::Primitive;
use rug::Float;

/// `|x|` at or above which the asymptotic expansion is used instead of the
/// Maclaurin series.  See the module docs for why it sits here.
pub const ASYMPTOTIC_FROM: f64 = 6.0;

/// Precision the `f64` entry points ask the kernels for: 53 bits of result
/// plus enough slack that the enclosure radius stays far below one ulp.
const F64_PREC: u32 = 96;

/// Hard cap on series terms.  The `|x| < 6` series needs ~44 and the
/// asymptotic one turns around after ~57; the cap only exists so that a
/// pathological input cannot spin.
const MAX_TERMS: usize = 400;

// ---------------------------------------------------------------------------
// f64 entry points
// ---------------------------------------------------------------------------

/// Fresnel sine integral `S(x) = ∫₀ˣ sin(π t²/2) dt`.
///
/// `None` only for a non-finite argument: `S` is entire, and real on all of
/// the reals.
pub fn fresnel_s(x: f64) -> Option<f64> {
    fresnel_pair_f64(x).map(|(s, _)| s)
}

/// Fresnel cosine integral `C(x) = ∫₀ˣ cos(π t²/2) dt`.
///
/// `None` only for a non-finite argument.
pub fn fresnel_c(x: f64) -> Option<f64> {
    fresnel_pair_f64(x).map(|(_, c)| c)
}

fn fresnel_pair_f64(x: f64) -> Option<(f64, f64)> {
    if !x.is_finite() {
        return None;
    }
    let (s, c) = fresnel_pair_ball(&Float::with_val(F64_PREC, x), F64_PREC)?;
    Some((s.mid.to_f64(), c.mid.to_f64()))
}

// ---------------------------------------------------------------------------
// Rigorous point kernel
// ---------------------------------------------------------------------------

/// Enclosures of `(S(x), C(x))` at `prec` bits.
///
/// The returned balls *contain* the true values: every truncation and every
/// rounding in the sums below is charged to the radius.  `None` for a
/// non-finite `x`.
pub fn fresnel_pair_ball(x: &Float, prec: u32) -> Option<(ArbBall, ArbBall)> {
    if !x.is_finite() {
        return None;
    }
    let work = x.prec().max(prec);
    let neg = x.is_sign_negative();
    let ax = Float::with_val(work, x.abs_ref());
    let (mut s, mut c) = if ax.to_f64() < ASYMPTOTIC_FROM {
        fresnel_series_ball(&ax, prec)?
    } else {
        fresnel_asymptotic_ball(&ax, prec)?
    };
    if neg {
        s.mid = -s.mid;
        c.mid = -c.mid;
    }
    Some((s, c))
}

/// Round a working-precision value plus an absolute error bound outward into a
/// ball at `prec`.
fn ball_from(value: &Float, abs_err: &Float, prec: u32) -> ArbBall {
    let work = value.prec().max(prec) + 32;
    let mid = Float::with_val(prec, value);
    // `value` carries more precision than `mid`; that truncation is part of
    // the enclosure, not a free lunch.
    let trunc = Float::with_val(prec, Float::with_val(work, value - &mid).abs());
    let mut rad = Float::with_val(prec, abs_err.abs_ref()) + trunc;
    // One ulp of `mid` for the roundings performed in this function itself.
    let mut bump = Float::with_val(prec, mid.abs_ref());
    bump >>= prec.saturating_sub(2);
    rad += bump;
    ArbBall { mid, rad, prec }
}

/// Maclaurin series, for `0 ≤ x < ASYMPTOTIC_FROM`.
///
/// With `a = (π/2)x²`, `uₙ = a^{2n}/(2n)!` and `vₙ = a^{2n+1}/(2n+1)!`:
///
/// ```text
/// C(x) = x·Σ_{n≥0} (−1)ⁿ uₙ/(4n+1),      S(x) = x·Σ_{n≥0} (−1)ⁿ vₙ/(4n+3)
/// ```
///
/// which is DLMF 7.6.4 / 7.6.6 with the `π/2` normalisation folded into `a`.
///
/// **Truncation.**  The series alternates, and once `a² < (2n+1)(2n+2)` its
/// terms decrease monotonically; from that point the Leibniz bound applies and
/// the tail is at most the first omitted term.  The loop therefore stops only
/// when *both* "terms are decreasing" and "the next term is below `2⁻ʷ⁺⁸`"
/// hold — the first condition alone would let it stop while the terms are
/// still growing, and the second alone would not license the Leibniz bound.
///
/// **Rounding.**  Each term is reached from the previous by a bounded number
/// of correctly-rounded MPFR operations, so its relative error after `N` steps
/// is at most `2N·2⁻ʷ`; summing `N` such terms, each no larger than the
/// largest term `U`, bounds the accumulated rounding by `4N²·U·2⁻ʷ`.  `w` is
/// chosen `64 + 2.3x²` bits above the target precision, so with `N ≤ 64` that
/// budget is spent many orders below the requested accuracy.
fn fresnel_series_ball(x: &Float, prec: u32) -> Option<(ArbBall, ArbBall)> {
    let xf = x.to_f64();
    // The largest term is ≈ e^{(π/2)x²}, i.e. 2.27·x² bits of cancellation.
    let cancel = (2.3 * xf * xf).ceil().max(0.0) as u32;
    let w = prec + 64 + cancel;

    let pi = Float::with_val(w, rug::float::Constant::Pi);
    let x2 = Float::with_val(w, x * x);
    let a = Float::with_val(w, Float::with_val(w, &pi * &x2) / 2u32);
    let a2 = Float::with_val(w, &a * &a);

    let mut u = Float::with_val(w, 1); // a^{2n}/(2n)!
    let mut v = Float::with_val(w, &a); // a^{2n+1}/(2n+1)!
    let mut sum_c = Float::new(w);
    let mut sum_s = Float::new(w);
    let mut biggest = Float::with_val(w, 1);
    let mut used = 0usize;
    let mut trunc = Float::with_val(w, f64::INFINITY);
    let tiny_threshold = Float::with_val(w, 1.0) >> (w - 8);

    for n in 0..MAX_TERMS {
        used = n + 1;
        let nn = n as u32;
        let tc = Float::with_val(w, &u / (4 * nn + 1));
        let ts = Float::with_val(w, &v / (4 * nn + 3));
        if n % 2 == 0 {
            sum_c += &tc;
            sum_s += &ts;
        } else {
            sum_c -= &tc;
            sum_s -= &ts;
        }
        let mag = Float::with_val(w, u.abs_ref()).max(&Float::with_val(w, v.abs_ref()));
        if mag > biggest {
            biggest = mag;
        }
        // uₙ₊₁ = uₙ·a²/((2n+1)(2n+2)),  vₙ₊₁ = vₙ·a²/((2n+2)(2n+3)).
        let du = Float::with_val(w, f64::from(2 * nn + 1) * f64::from(2 * nn + 2));
        let dv = Float::with_val(w, f64::from(2 * nn + 2) * f64::from(2 * nn + 3));
        u *= &a2;
        u /= &du;
        v *= &a2;
        v /= &dv;
        let next = Float::with_val(w, u.abs_ref()).max(&Float::with_val(w, v.abs_ref()));
        if a2 < du && next < tiny_threshold {
            trunc = next;
            break;
        }
    }
    if !trunc.is_finite() {
        return None;
    }

    let mut round = Float::with_val(w, &biggest * Float::with_val(w, (4 * used * used) as f64));
    round >>= w;
    // S and C are `x` times the sums, so the error scales with |x| too.
    let scale = Float::with_val(w, x.abs_ref()).max(&Float::with_val(w, 1.0));
    let err = Float::with_val(w, Float::with_val(w, &trunc + &round) * &scale);

    let s_val = Float::with_val(w, x * &sum_s);
    let c_val = Float::with_val(w, x * &sum_c);
    Some((ball_from(&s_val, &err, prec), ball_from(&c_val, &err, prec)))
}

/// Asymptotic expansion, for `x ≥ ASYMPTOTIC_FROM`.
///
/// Repeated integration by parts of `∫ₓ^∞ e^{iπt²/2} dt` gives, with
/// `θ = πx²/2`,
///
/// ```text
/// C(x) = 1/2 + f(x)·sin θ − g(x)·cos θ,
/// S(x) = 1/2 − f(x)·cos θ − g(x)·sin θ,
/// f(x) = (1/π)·Σ_{m≥0} (−1)^m t_{2m},   g(x) = (1/π)·Σ_{m≥0} (−1)^m t_{2m+1},
/// t_n = (2n−1)!!/(πⁿ x^{2n+1}),         t_{n+1} = t_n·(2n+1)/(π x²).
/// ```
///
/// This is DLMF 7.12.1–7.12.3 written as one sequence: there `(1/2)_{2m}` is
/// `(4m−1)!!/2^{2m}` and `(πz²/2)^{2m}` is `(πz²)^{2m}/2^{2m}`, so the powers
/// of two cancel and `f` reduces exactly to the even-indexed `t_n`.
///
/// **Truncation is rigorous, not heuristic.**  DLMF §7.12(ii): for
/// `|ph z| ≤ π/8` — in particular on the positive real axis — `R_n^{(f)}` and
/// `R_n^{(g)}` "are bounded in magnitude by the first neglected terms … and
/// have the same signs as these terms when `ph z = 0`".  The loop stops where
/// the terms turn around and charges the next unused term of *each* series to
/// the radius.
///
/// `θ` is formed at `prec + 96 + log₂x` bits because `sin θ` inherits the
/// **absolute** error of `θ`, which grows like `x²`: at `x = 100` an `f64` `θ`
/// is already wrong in the 12th digit and `S` would silently follow it.
fn fresnel_asymptotic_ball(x: &Float, prec: u32) -> Option<(ArbBall, ArbBall)> {
    let xf = x.to_f64();
    let grow = if xf > 1.0 {
        xf.log2().ceil().max(0.0) as u32
    } else {
        0
    };
    let w = prec + 96 + grow;

    let pi = Float::with_val(w, rug::float::Constant::Pi);
    let x2 = Float::with_val(w, x * x);
    let pix2 = Float::with_val(w, &pi * &x2);
    let theta = Float::with_val(w, &pix2 / 2u32);
    let (sin_t, cos_t) = Float::with_val(w, &theta).sin_cos(Float::new(w));

    let mut t = Float::with_val(w, Float::with_val(w, 1) / x); // t₀ = 1/x
    let mut sum_a = Float::new(w); // Σ (−1)^m t_{2m}
    let mut sum_b = Float::new(w); // Σ (−1)^m t_{2m+1}
    let mut err_a = Float::with_val(w, f64::INFINITY);
    let mut err_b = Float::with_val(w, f64::INFINITY);
    let mut used = 0usize;

    for n in 0..MAX_TERMS {
        used = n + 1;
        let m = n / 2;
        let plus = m % 2 == 0;
        if n % 2 == 0 {
            if plus {
                sum_a += &t;
            } else {
                sum_a -= &t;
            }
        } else if plus {
            sum_b += &t;
        } else {
            sum_b -= &t;
        }
        let step = Float::with_val(w, f64::from(2 * (n as u32) + 1));
        let next = Float::with_val(w, Float::with_val(w, &t * &step) / &pix2);
        if next >= t || n + 1 == MAX_TERMS {
            // The terms have turned around: charge the next unused term of
            // each of the two interleaved series.
            let step2 = Float::with_val(w, f64::from(2 * (n as u32) + 3));
            let after = Float::with_val(w, Float::with_val(w, &next * &step2) / &pix2);
            if n % 2 == 0 {
                err_b = next;
                err_a = after;
            } else {
                err_a = next;
                err_b = after;
            }
            break;
        }
        t = next;
    }
    if !err_a.is_finite() || !err_b.is_finite() {
        return None;
    }

    let f = Float::with_val(w, &sum_a / &pi);
    let g = Float::with_val(w, &sum_b / &pi);
    let ef = Float::with_val(w, &err_a / &pi);
    let eg = Float::with_val(w, &err_b / &pi);

    let half = Float::with_val(w, 0.5);
    let c_val = Float::with_val(
        w,
        &half
            + Float::with_val(
                w,
                Float::with_val(w, &f * &sin_t) - Float::with_val(w, &g * &cos_t),
            ),
    );
    let s_val = Float::with_val(
        w,
        &half
            - Float::with_val(
                w,
                Float::with_val(w, &f * &cos_t) + Float::with_val(w, &g * &sin_t),
            ),
    );

    // |ΔC| ≤ |Δf| + |Δg| + (|f|+|g|)·|Δθ| + rounding, with |Δθ| ≤ θ·2⁻ʷ and
    // the same `4N²` rounding budget as the power series.
    let mut dtheta = Float::with_val(w, &theta);
    dtheta >>= w;
    let amp = Float::with_val(
        w,
        Float::with_val(w, f.abs_ref()) + Float::with_val(w, g.abs_ref()),
    );
    let mut round = Float::with_val(w, (4 * used * used) as f64);
    round >>= w;
    let err = Float::with_val(
        w,
        Float::with_val(w, &ef + &eg)
            + Float::with_val(w, Float::with_val(w, &amp * &dtheta) + round),
    );
    Some((ball_from(&s_val, &err, prec), ball_from(&c_val, &err, prec)))
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// `π/2` as a pool literal — the constant in `S′(x) = sin(πx²/2)`.
///
/// A `Float` literal, not the symbol `pi`: a derivative carrying a free symbol
/// would make `diff(fresnels(x), x)` an expression with an *unbound* variable,
/// which every numeric consumer — the verification gate included — would then
/// refuse, or worse treat as a second dimension.  `erf` already makes the same
/// choice for `2/√π`.
fn half_pi(pool: &ExprPool) -> ExprId {
    pool.float(std::f64::consts::FRAC_PI_2, 53)
}

/// `sin(πx²/2)` or `cos(πx²/2)`.
fn phase(trig: &'static str, x: ExprId, pool: &ExprPool) -> ExprId {
    let x2 = pool.pow(x, pool.integer(2_i32));
    let arg = pool.mul(vec![half_pi(pool), x2]);
    pool.func(trig, vec![arg])
}

fn unary_ball(args: &[ArbBall]) -> Option<&ArbBall> {
    match args {
        [only] => Some(only),
        _ => None,
    }
}

/// `S(x) = ∫₀ˣ sin(πt²/2) dt`.  See the module docs for the convention.
pub struct FresnelSPrimitive;

impl Primitive for FresnelSPrimitive {
    fn name(&self) -> &'static str {
        "fresnels"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        format!("S({})", pool.display(args[0]))
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        if args.len() != 1 {
            return None;
        }
        let x = args[0];
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        Some(pool.mul(vec![phase("sin", x, pool), dx]))
    }

    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        if args.len() != 1 {
            return None;
        }
        Some(vec![pool.mul(vec![cotan, phase("sin", args[0], pool)])])
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [x] => fresnel_s(*x),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        unary_ball(args)?.fresnel_s()
    }
}

/// `C(x) = ∫₀ˣ cos(πt²/2) dt`.  See the module docs for the convention.
pub struct FresnelCPrimitive;

impl Primitive for FresnelCPrimitive {
    fn name(&self) -> &'static str {
        "fresnelc"
    }

    fn pretty(&self, args: &[ExprId], pool: &ExprPool) -> String {
        format!("C({})", pool.display(args[0]))
    }

    fn diff_forward(&self, args: &[ExprId], wrt: ExprId, pool: &ExprPool) -> Option<ExprId> {
        if args.len() != 1 {
            return None;
        }
        let x = args[0];
        let dx = crate::diff::diff(x, wrt, pool).ok()?.value;
        Some(pool.mul(vec![phase("cos", x, pool), dx]))
    }

    fn diff_reverse(&self, args: &[ExprId], cotan: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
        if args.len() != 1 {
            return None;
        }
        Some(vec![pool.mul(vec![cotan, phase("cos", args[0], pool)])])
    }

    fn numeric_f64(&self, args: &[f64]) -> Option<f64> {
        match args {
            [x] => fresnel_c(*x),
            _ => None,
        }
    }

    fn numeric_ball(&self, args: &[ArbBall]) -> Option<ArbBall> {
        unary_ball(args)?.fresnel_c()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(x, S(x), C(x))`.
    ///
    /// Sources, in the normalised π/2 convention:
    ///
    /// * Abramowitz & Stegun, *Handbook of Mathematical Functions*, Table 7.7
    ///   (§7.3) tabulates `C(x)`, `S(x)` to 7D; the entries at
    ///   `x = 0.5, 1.0, 1.5, 2.0, 3.0` below agree with it to every printed
    ///   digit (e.g. A&S gives `C(1.0) = 0.7798934`, `S(1.0) = 0.4382591`).
    /// * The full 16-digit values are from `scipy.special.fresnel` (SciPy
    ///   1.18) — an *independent* implementation (Cephes rational
    ///   approximations), not the series below, computed offline; SciPy is not
    ///   a dependency of this crate.
    const REFERENCE: &[(f64, f64, f64)] = &[
        (0.0, 0.0, 0.0),
        (0.25, 0.008_175_600_235_777_757, 0.249_759_150_356_543_2),
        (0.5, 0.064_732_432_859_999_29, 0.492_344_225_871_446_4),
        (1.0, 0.438_259_147_390_354_7, 0.779_893_400_376_823),
        (1.5, 0.697_504_960_082_093, 0.445_261_176_039_821_57),
        (2.0, 0.343_415_678_363_698_24, 0.488_253_406_075_340_73),
        (2.5, 0.619_181_755_819_592_9, 0.457_413_009_641_777_06),
        (3.0, 0.496_312_998_967_375, 0.605_720_789_297_685_7),
        (4.0, 0.420_515_754_246_928_44, 0.498_426_033_038_177_6),
        (5.0, 0.499_191_381_917_116_87, 0.563_631_188_704_012_2),
        (8.0, 0.460_214_214_393_014_46, 0.499_802_180_377_197_15),
        (20.0, 0.484_084_535_925_953_9, 0.499_987_334_972_344_4),
        (100.0, 0.496_816_901_147_837_55, 0.499_999_898_678_817_9),
    ];

    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            (got - want).abs() / want.abs()
        }
    }

    #[test]
    fn matches_published_reference_values() {
        for &(x, s, c) in REFERENCE {
            let gs = fresnel_s(x).unwrap();
            let gc = fresnel_c(x).unwrap();
            assert!(rel(gs, s) < 1e-14, "S({x}): got {gs}, want {s}");
            assert!(rel(gc, c) < 1e-14, "C({x}): got {gc}, want {c}");
        }
    }

    /// The one test that would fail loudly if the unnormalised convention ever
    /// crept in: there `S(∞) = C(∞) = √(π/8) = 0.6267`, not `1/2`.
    #[test]
    fn half_at_infinity_pins_the_convention() {
        for x in [1.0e3_f64, 1.0e6, 1.0e9] {
            assert!((fresnel_s(x).unwrap() - 0.5).abs() < 1.0 / x);
            assert!((fresnel_c(x).unwrap() - 0.5).abs() < 1.0 / x);
        }
        // …and the approach really is 1/2, not 0.6267.
        assert!((fresnel_s(1.0e9).unwrap() - 0.5).abs() < 1e-9);
        assert!((fresnel_c(1.0e9).unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn both_functions_are_odd() {
        for x in [0.3_f64, 1.7, 4.5, 6.5, 30.0] {
            assert_eq!(fresnel_s(-x).unwrap(), -fresnel_s(x).unwrap());
            assert_eq!(fresnel_c(-x).unwrap(), -fresnel_c(x).unwrap());
        }
    }

    /// A discontinuity at a series/asymptotic seam is the classic bug in this
    /// kind of code, so walk across `ASYMPTOTIC_FROM` in both directions and
    /// require the two *different* algorithms to agree to `f64` resolution.
    #[test]
    fn the_two_regimes_agree_across_the_switchover() {
        let e = ASYMPTOTIC_FROM;
        for d in [1e-9_f64, 1e-7, 1e-5, 1e-3] {
            let (s_lo, c_lo) = (fresnel_s(e - d).unwrap(), fresnel_c(e - d).unwrap());
            let (s_hi, c_hi) = (fresnel_s(e + d).unwrap(), fresnel_c(e + d).unwrap());
            // |S′| ≤ 1 and |C′| ≤ 1, so the true values differ by at most 2d.
            assert!((s_hi - s_lo).abs() <= 2.0 * d + 1e-14, "S seam at ±{d}");
            assert!((c_hi - c_lo).abs() <= 2.0 * d + 1e-14, "C seam at ±{d}");
        }
        // Directly: both kernels evaluated at the seam itself.
        let series = fresnel_series_ball(&Float::with_val(F64_PREC, ASYMPTOTIC_FROM), F64_PREC)
            .expect("series at the seam");
        let asym = fresnel_asymptotic_ball(&Float::with_val(F64_PREC, ASYMPTOTIC_FROM), F64_PREC)
            .expect("asymptotic at the seam");
        assert!(
            (series.0.mid.to_f64() - asym.0.mid.to_f64()).abs() < 1e-15,
            "S: series {} vs asymptotic {}",
            series.0.mid.to_f64(),
            asym.0.mid.to_f64()
        );
        assert!(
            (series.1.mid.to_f64() - asym.1.mid.to_f64()).abs() < 1e-15,
            "C: series {} vs asymptotic {}",
            series.1.mid.to_f64(),
            asym.1.mid.to_f64()
        );
    }

    /// The claimed enclosure has to actually enclose.  Two checks, because
    /// neither alone is enough:
    ///
    /// * a *low*-precision ball must contain the high-precision midpoint —
    ///   this is the real enclosure property, and it would fail immediately if
    ///   the truncation or rounding budget were understated;
    /// * the high-precision midpoint must agree with the published table to
    ///   `f64` resolution — the table itself is only good to ~10⁻¹⁶, so it
    ///   cannot be asked to sit inside a 10⁻⁵⁰ ball.
    #[test]
    fn the_enclosures_enclose() {
        for &(x, s, c) in REFERENCE {
            let (hs, hc) = fresnel_pair_ball(&Float::with_val(200, x), 200).unwrap();
            assert!((hs.mid.to_f64() - s).abs() < 1e-15, "S({x}) vs table");
            assert!((hc.mid.to_f64() - c).abs() < 1e-15, "C({x}) vs table");
            assert!(hs.rad.to_f64() < 1e-30, "S({x}) radius {}", hs.rad);
            assert!(hc.rad.to_f64() < 1e-30, "C({x}) radius {}", hc.rad);

            for prec in [32_u32, 53, 96] {
                let (ls, lc) = fresnel_pair_ball(&Float::with_val(prec, x), prec).unwrap();
                // Compared as `Float`s: rounding the 200-bit value to `f64`
                // first would inject 10⁻¹⁹ of error into a 10⁻³¹ ball.
                assert!(
                    ls.lo() <= hs.mid && hs.mid <= ls.hi(),
                    "S({x}) at {prec} bits: {} outside {ls}",
                    hs.mid
                );
                assert!(
                    lc.lo() <= hc.mid && hc.mid <= lc.hi(),
                    "C({x}) at {prec} bits: {} outside {lc}",
                    hc.mid
                );
            }
        }
    }

    /// `S′ = sin(πx²/2)` and `C′ = cos(πx²/2)` checked against the kernels
    /// themselves by central differences — the property the verification gate
    /// will rely on, tested numerically rather than symbolically.
    #[test]
    fn the_derivatives_are_the_integrands() {
        let h = 1e-5_f64;
        for x in [0.3_f64, 1.0, 2.4, 5.0] {
            let ds = (fresnel_s(x + h).unwrap() - fresnel_s(x - h).unwrap()) / (2.0 * h);
            let dc = (fresnel_c(x + h).unwrap() - fresnel_c(x - h).unwrap()) / (2.0 * h);
            let t = std::f64::consts::FRAC_PI_2 * x * x;
            assert!((ds - t.sin()).abs() < 1e-6, "S′({x}): {ds} vs {}", t.sin());
            assert!((dc - t.cos()).abs() < 1e-6, "C′({x}): {dc} vs {}", t.cos());
        }
    }

    #[test]
    fn non_finite_arguments_are_declined() {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(fresnel_s(x).is_none());
            assert!(fresnel_c(x).is_none());
        }
    }
}
