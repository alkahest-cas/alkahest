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
}
