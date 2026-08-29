//! Real special-function kernels shared by the primitive registry and ball
//! arithmetic.

use rug::Float;

/// Euler–Mascheroni constant γ (53-bit).
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Lower domain endpoint for the principal Lambert branch: `−1/e`.
pub fn lambert_w0_domain_min() -> f64 {
    -std::f64::consts::E.recip()
}

/// Principal-branch Lambert W₀(x), defined for `x ≥ −1/e`.
pub fn lambert_w0(x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    let em = lambert_w0_domain_min();
    if x < em - 1e-15 {
        return None;
    }
    if x == 0.0 {
        return Some(0.0);
    }
    if (x - em).abs() < 1e-15 {
        return Some(-1.0);
    }

    let mut w = if x < 0.0 {
        let p = (2.0 * (x - em)).max(0.0).sqrt();
        -1.0 + p * (1.0 - p / 3.0)
    } else if x <= 1.0 {
        x
    } else {
        let l1 = x.ln();
        l1 - l1.ln()
    };

    for _ in 0..64 {
        let ew = w.exp();
        let f = w * ew - x;
        if f.abs() < 1e-15 * x.abs().max(1.0) {
            return Some(w);
        }
        let fp = ew * (w + 1.0);
        let fpp = ew * (2.0 + w);
        let denom = 2.0 * fp * fp - f * fpp;
        w = if denom.abs() < 1e-300 {
            w - f / fp
        } else {
            w - 2.0 * f * fp / denom
        };
        if !w.is_finite() {
            return None;
        }
    }
    Some(w)
}

/// Digamma ψ(x).  Returns `None` at non-positive integer poles.
pub fn digamma(x: f64) -> Option<f64> {
    if x.is_nan() {
        return None;
    }
    if x <= 0.0 && x.fract() == 0.0 {
        return None;
    }
    let mut f = Float::with_val(53, x);
    f.digamma_mut();
    Some(f.to_f64())
}

/// Bessel J₀(x).
pub fn bessel_j0(x: f64) -> f64 {
    Float::with_val(53, x).jn(0).to_f64()
}

/// Bessel J₁(x).
pub fn bessel_j1(x: f64) -> f64 {
    Float::with_val(53, x).jn(1).to_f64()
}

// ---------------------------------------------------------------------------
// Trigamma ψ₁ = ψ′ (3.10.0)
// ---------------------------------------------------------------------------

/// `B₂, B₄, …, B₃₀` as exact `(numerator, denominator)` pairs.
///
/// Written as rationals rather than `f64` so that the tail below stays exact
/// at whatever working precision the caller asked for; `B₃₀ = 8615841276005/14322`
/// would already be inexact as a `double`.
const BERNOULLI_2N: [(i64, i64); 15] = [
    (1, 6),
    (-1, 30),
    (1, 42),
    (-1, 30),
    (5, 66),
    (-691, 2730),
    (7, 6),
    (-3617, 510),
    (43867, 798),
    (-174611, 330),
    (854513, 138),
    (-236364091, 2730),
    (8553103, 6),
    (-23749461029, 870),
    (8615841276005, 14322),
];

/// Trigamma `ψ₁(x) = ψ′(x) = Σ_{k≥0} (x+k)⁻²`, at the precision of `x`.
///
/// `None` at the double poles (`x = 0, −1, −2, …`) and for a non-finite `x`.
///
/// # Method
///
/// Recurrence up to a large argument, then Euler–Maclaurin:
///
/// ```text
/// ψ₁(x) = Σ_{k=0}^{N−1} (x+k)⁻²  +  ψ₁(z),      z = x + N,
/// ψ₁(z) ~ 1/z + 1/(2z²) + Σ_{j≥1} B_{2j}·z^{−2j−1}.
/// ```
///
/// The shift target `N` is chosen from the working precision so that the first
/// omitted term, `≈ |B₃₂|/z³³`, is already below it — `z ≈ 136` at 200 bits,
/// `z ≈ 20` at 53.  The tail loop additionally stops as soon as a term fails to
/// decrease, because the expansion is asymptotic and eventually diverges; the
/// two conditions together are what keep this honest at high precision rather
/// than silently plateauing.
///
/// Negative arguments go through the reflection formula
/// `ψ₁(x) + ψ₁(1−x) = π²/sin²(πx)` (Abramowitz & Stegun 6.4.7), which maps
/// `x < 0` to `1 − x > 1`.
pub fn trigamma(x: &Float) -> Option<Float> {
    if !x.is_finite() {
        return None;
    }
    let prec = x.prec();
    let w = prec + 32;
    if *x <= 0u32 {
        // Poles at the non-positive integers.
        if x.is_integer() {
            return None;
        }
        let pi = Float::with_val(w, rug::float::Constant::Pi);
        let s = Float::with_val(w, Float::with_val(w, &pi * x).sin());
        if s.is_zero() {
            return None;
        }
        let refl = Float::with_val(
            w,
            Float::with_val(w, &pi * &pi) / Float::with_val(w, &s * &s),
        );
        let one_minus = Float::with_val(w, Float::with_val(w, 1u32) - x);
        let other = trigamma(&one_minus)?;
        return Some(Float::with_val(prec, refl - other));
    }

    // Shift target: the first omitted Euler–Maclaurin term is ≈ |B₃₂|/z³³ with
    // |B₃₂| < 2³⁴, so `z ≥ 2^((w+34)/33)` puts it under 2⁻ʷ.
    let target = f64::from(w + 34) / 33.0;
    let target = target.exp2().max(20.0);
    let mut z = Float::with_val(w, x);
    let mut head = Float::new(w);
    let mut guard = 0u32;
    while z.to_f64() < target {
        let inv = Float::with_val(w, Float::with_val(w, 1u32) / &z);
        head += Float::with_val(w, &inv * &inv);
        z += 1u32;
        guard += 1;
        if guard > 1_000_000 {
            return None;
        }
    }

    let zinv = Float::with_val(w, Float::with_val(w, 1u32) / &z);
    let zinv2 = Float::with_val(w, &zinv * &zinv);
    let mut acc = Float::with_val(w, &zinv + Float::with_val(w, &zinv2 / 2u32));
    // termⱼ = B_{2j}·z^{−2j−1}: start at j = 1 with z⁻³ and multiply by z⁻².
    let mut pw = Float::with_val(w, &zinv2 * &zinv);
    let mut prev = Float::with_val(w, f64::INFINITY);
    for &(num, den) in BERNOULLI_2N.iter() {
        let b = Float::with_val(w, num) / Float::with_val(w, den);
        let term = Float::with_val(w, b * &pw);
        let mag = Float::with_val(w, term.abs_ref());
        if mag >= prev {
            break;
        }
        acc += &term;
        prev = mag;
        pw = Float::with_val(w, &pw * &zinv2);
    }
    Some(Float::with_val(prec, head + acc))
}

/// Trigamma at `f64` precision.  `None` at the poles `x = 0, −1, −2, …`.
pub fn trigamma_f64(x: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    trigamma(&Float::with_val(96, x)).map(|v| v.to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lambert_w0_at_one() {
        assert!(lambert_w0(1.0).is_some());
    }

    #[test]
    fn lambert_w0_special_values() {
        assert_eq!(lambert_w0(0.0), Some(0.0));
        assert!((lambert_w0(lambert_w0_domain_min()).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn digamma_integer_values() {
        let psi1 = digamma(1.0).unwrap();
        assert!((psi1 + EULER_GAMMA).abs() < 1e-12);
        let psi2 = digamma(2.0).unwrap();
        assert!((psi2 - (1.0 - EULER_GAMMA)).abs() < 1e-12);
    }

    #[test]
    fn bessel_j0_at_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-12);
    }

    /// Closed forms from Abramowitz & Stegun §6.4: `ψ₁(1) = π²/6` (6.4.2 at
    /// `n = 1`), `ψ₁(1/2) = π²/2` (6.4.4), and the recurrence
    /// `ψ₁(x+1) = ψ₁(x) − 1/x²` (6.4.6).
    #[test]
    fn trigamma_closed_forms() {
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        let got = trigamma_f64(1.0).unwrap();
        assert!((got - pi2 / 6.0).abs() < 1e-14, "ψ₁(1) = {got}");
        let got = trigamma_f64(0.5).unwrap();
        assert!((got - pi2 / 2.0).abs() < 1e-13, "ψ₁(1/2) = {got}");
        let got = trigamma_f64(2.0).unwrap();
        assert!((got - (pi2 / 6.0 - 1.0)).abs() < 1e-14, "ψ₁(2) = {got}");
    }

    #[test]
    fn trigamma_satisfies_its_recurrence() {
        for x in [0.3_f64, 1.0, 2.5, 7.75, 40.0, 1e4] {
            let a = trigamma_f64(x + 1.0).unwrap();
            let b = trigamma_f64(x).unwrap() - 1.0 / (x * x);
            assert!(
                (a - b).abs() < 1e-13 * a.abs().max(1.0),
                "ψ₁({x}+1) = {a} vs ψ₁({x}) − 1/x² = {b}"
            );
        }
    }

    /// The reflection formula `ψ₁(x) + ψ₁(1−x) = π²/sin²(πx)` (A&S 6.4.7) is
    /// what covers the negative axis; check it *as an identity* rather than
    /// against the branch that implements it.
    #[test]
    fn trigamma_reflection_holds_on_the_negative_axis() {
        let pi = std::f64::consts::PI;
        for x in [-0.25_f64, -0.5, -1.5, -2.75, -10.3] {
            let lhs = trigamma_f64(x).unwrap() + trigamma_f64(1.0 - x).unwrap();
            let rhs = pi * pi / (pi * x).sin().powi(2);
            assert!(
                (lhs - rhs).abs() < 1e-10 * rhs.abs(),
                "reflection at {x}: {lhs} vs {rhs}"
            );
        }
    }

    /// `ψ₁` is the derivative of `ψ`, which is the entire reason it exists
    /// here.  Central differences of the *digamma* kernel, not of `ψ₁` itself.
    #[test]
    fn trigamma_is_the_derivative_of_digamma() {
        let h = 1e-6;
        for x in [0.4_f64, 1.0, 2.5, 9.0] {
            let fd = (digamma(x + h).unwrap() - digamma(x - h).unwrap()) / (2.0 * h);
            let got = trigamma_f64(x).unwrap();
            assert!(
                (got - fd).abs() < 1e-6 * got.abs(),
                "ψ₁({x}): {got} vs {fd}"
            );
        }
    }

    #[test]
    fn trigamma_declines_at_its_double_poles() {
        for pole in [0.0_f64, -1.0, -2.0, -17.0] {
            assert!(trigamma_f64(pole).is_none(), "{pole}");
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(trigamma_f64(bad).is_none());
        }
    }
}
