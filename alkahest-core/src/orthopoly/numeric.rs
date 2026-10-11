//! Numerical kernels: `f64` three-term recurrences for an integer degree, and
//! Arb's `arb_hypgeom_*` for rigorous balls and for non-integer degrees.
//!
//! # Which path answers
//!
//! * **`f64`, integer degree** (`|n| ≤ 10⁷`): the forward three-term
//!   recurrence of each family (DLMF 18.9.1–18.9.2, 14.10.3), after the same
//!   degree reflections the exact expansion uses.  Forward recurrence is the
//!   stable direction for all of these on and off the orthogonality interval
//!   (the polynomial is the dominant solution).  A non-finite result
//!   (`H_400(30)` overflows) is `None`, never `inf`.
//! * **`f64`, non-integer degree, and every ball**: Arb, when this build
//!   carries FLINT's Arb layer (`cfg(flint_arb)`); `None` otherwise.  The
//!   `f64` value is accepted only when Arb certifies at least 53 correct
//!   bits, retrying at 128 and 256 bits of working precision.
//!
//! The associated Legendre function is the **Ferrers** function (Arb's
//! `type = 0`), real only on `[−1, 1]` for odd order; outside that interval
//! with odd `m` both kernels decline rather than return a real part.

use crate::ball::ArbBall;

use super::OrthoFamily;

/// Largest `|n|` the `f64` recurrence will run (`O(n)` steps).
const MAX_RECURRENCE_DEGREE: f64 = 1e7;

fn as_int(v: f64) -> Option<i64> {
    if v.is_finite() && v == v.trunc() && v.abs() <= MAX_RECURRENCE_DEGREE {
        Some(v as i64)
    } else {
        None
    }
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

/// Evaluate `family(args…)` in `f64`.  `args` = `[n, params…, x]`.
pub(crate) fn eval_f64(family: OrthoFamily, args: &[f64]) -> Option<f64> {
    if args.len() != family.arity() || args.iter().any(|a| !a.is_finite()) {
        return None;
    }
    if let Some(v) = integer_degree_f64(family, args) {
        return finite(v);
    }
    arb_f64(family, args)
}

/// The recurrence path; `None` when the degree (or Legendre order) is not an
/// admissible integer for it, so the caller can try Arb.
fn integer_degree_f64(family: OrthoFamily, args: &[f64]) -> Option<f64> {
    let n = as_int(args[0])?;
    let x = *args.last().unwrap();
    let (n, sign) = match family {
        OrthoFamily::Legendre | OrthoFamily::AssocLegendre if n < 0 => (-n - 1, 1.0),
        OrthoFamily::ChebyshevT if n < 0 => (-n, 1.0),
        OrthoFamily::ChebyshevU if n == -1 => return Some(0.0),
        OrthoFamily::ChebyshevU if n < -1 => (-n - 2, -1.0),
        _ if n < 0 => return None,
        _ => (n, 1.0),
    };
    let v = match family {
        OrthoFamily::Legendre => legendre(n, x),
        OrthoFamily::AssocLegendre => assoc_legendre(n, as_int(args[1])?, x)?,
        OrthoFamily::HermiteH => {
            three_term(n, 1.0, 2.0 * x, |k, p1, p0| 2.0 * x * p1 - 2.0 * k * p0)
        }
        OrthoFamily::HermiteHe => three_term(n, 1.0, x, |k, p1, p0| x * p1 - k * p0),
        OrthoFamily::Laguerre => laguerre(n, 0.0, x),
        OrthoFamily::AssocLaguerre => laguerre(n, args[1], x),
        OrthoFamily::ChebyshevT => three_term(n, 1.0, x, |_, p1, p0| 2.0 * x * p1 - p0),
        OrthoFamily::ChebyshevU => three_term(n, 1.0, 2.0 * x, |_, p1, p0| 2.0 * x * p1 - p0),
        OrthoFamily::Gegenbauer => {
            let a = args[1];
            three_term(n, 1.0, 2.0 * a * x, |k, p1, p0| {
                (2.0 * (k + a) * x * p1 - (k + 2.0 * a - 1.0) * p0) / (k + 1.0)
            })
        }
        OrthoFamily::Jacobi => jacobi(n, args[1], args[2], x),
    };
    Some(sign * v)
}

/// `p_0 = a0`, `p_1 = a1`, `p_{k+1} = step(k, p_k, p_{k−1})` for `k ≥ 1`.
fn three_term(n: i64, a0: f64, a1: f64, step: impl Fn(f64, f64, f64) -> f64) -> f64 {
    if n == 0 {
        return a0;
    }
    let (mut p0, mut p1) = (a0, a1);
    for k in 1..n {
        let p2 = step(k as f64, p1, p0);
        p0 = p1;
        p1 = p2;
        if !p1.is_finite() {
            return p1;
        }
    }
    p1
}

fn legendre(n: i64, x: f64) -> f64 {
    three_term(n, 1.0, x, |k, p1, p0| {
        ((2.0 * k + 1.0) * x * p1 - k * p0) / (k + 1.0)
    })
}

fn laguerre(n: i64, alpha: f64, x: f64) -> f64 {
    three_term(n, 1.0, 1.0 + alpha - x, |k, p1, p0| {
        ((2.0 * k + 1.0 + alpha - x) * p1 - (k + alpha) * p0) / (k + 1.0)
    })
}

/// Ferrers `P_n^m(x)` with the Condon–Shortley phase, `n ≥ 0`.
fn assoc_legendre(n: i64, m: i64, x: f64) -> Option<f64> {
    let am = m.unsigned_abs() as i64;
    if am > n {
        return Some(0.0);
    }
    let one_minus_x2 = (1.0 - x) * (1.0 + x);
    // P_m^m = (−1)^m (2m−1)!! (1−x²)^{m/2}
    let pmm = if am == 0 {
        1.0
    } else if x.abs() <= 1.0 {
        let s = one_minus_x2.sqrt();
        let mut p = 1.0;
        let mut f = 1.0;
        for _ in 0..am {
            p *= -f * s;
            f += 2.0;
        }
        p
    } else if am % 2 == 0 {
        let mut dfact = 1.0;
        let mut f = 1.0;
        for _ in 0..am {
            dfact *= f;
            f += 2.0;
        }
        dfact * one_minus_x2.powi((am / 2) as i32)
    } else {
        // (1 − x²)^{m/2} is imaginary for odd m off [−1, 1].
        return None;
    };
    let mut v = pmm;
    if n > am {
        let mut p0 = pmm;
        let mut p1 = x * (2 * am + 1) as f64 * pmm;
        for l in (am + 2)..=n {
            let p2 = ((2 * l - 1) as f64 * x * p1 - (l + am - 1) as f64 * p0) / (l - am) as f64;
            p0 = p1;
            p1 = p2;
        }
        v = p1;
    }
    if m < 0 {
        // P_n^{−m} = (−1)^m (n−m)!/(n+m)! P_n^m
        let mut r = if am % 2 == 1 { -1.0 } else { 1.0 };
        for j in (n - am + 1)..=(n + am) {
            r /= j as f64;
        }
        v *= r;
    }
    Some(v)
}

/// Jacobi by the DLMF 18.9.2 recurrence, falling back to the explicit
/// hypergeometric sum when a recurrence denominator vanishes
/// (`n + a + b = 0` or `2n + a + b − 2 = 0` for some step).
fn jacobi(n: i64, a: f64, b: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut p0 = 1.0;
    let mut p1 = (a + 1.0) + (a + b + 2.0) * (x - 1.0) / 2.0;
    for k in 2..=n {
        let k = k as f64;
        let c = 2.0 * k + a + b;
        let a1 = 2.0 * k * (k + a + b) * (c - 2.0);
        if a1 == 0.0 {
            return jacobi_sum(n, a, b, x);
        }
        let a2 = (c - 1.0) * (a * a - b * b);
        let a3 = (c - 2.0) * (c - 1.0) * c;
        let a4 = 2.0 * (k + a - 1.0) * (k + b - 1.0) * c;
        let p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
        p0 = p1;
        p1 = p2;
    }
    p1
}

/// `Σ_k C(n,k)(−1)ᵏ (n+a+b+1)_k (a+k+1)_{n−k}/n! · ((1−x)/2)ᵏ` (DLMF 18.5.7).
///
/// The terms alternate and cancel, so the sum is carried in 256-bit MPFR
/// arithmetic (`a`, `b`, `x` are exact binary values) and rounded once.
fn jacobi_sum(n: i64, a: f64, b: f64, x: f64) -> f64 {
    use rug::Float;
    const P: u32 = 256;
    let a = Float::with_val(P, a);
    let b = Float::with_val(P, b);
    let t = Float::with_val(P, (Float::with_val(P, 1) - Float::with_val(P, x)) / 2u32);
    let s = Float::with_val(P, &a + &b) + (n + 1) as f64;
    let mut nfact = Float::with_val(P, 1);
    for j in 1..=n {
        nfact *= j as f64;
    }
    let mut total = Float::with_val(P, 0);
    let mut binom = Float::with_val(P, 1);
    let mut rising = Float::with_val(P, 1);
    let mut tk = Float::with_val(P, 1);
    for k in 0..=n {
        let mut tail = Float::with_val(P, 1);
        for j in (k + 1)..=n {
            tail *= Float::with_val(P, &a + j as f64);
        }
        let mut term = Float::with_val(P, &binom * &rising);
        term *= &tail;
        term *= &tk;
        if k % 2 == 1 {
            total -= term;
        } else {
            total += term;
        }
        binom *= (n - k) as f64;
        binom /= (k + 1) as f64;
        rising *= Float::with_val(P, &s + k as f64);
        tk *= &t;
    }
    (total / nfact).to_f64()
}

// ---------------------------------------------------------------------------
// Arb
// ---------------------------------------------------------------------------

#[cfg(flint_arb)]
mod arb_impl {
    use super::OrthoFamily;
    use crate::ball::ArbBall;
    use crate::flint::acb::Arb;
    use crate::flint::arb as ffi;
    use rug::float::Round;
    use rug::Float;

    fn available() -> bool {
        ffi::abi_self_check().is_ok()
    }

    /// `family(args)` over Arb balls at `prec` bits.  `args = [n, params…, x]`.
    fn call(family: OrthoFamily, args: &[Arb], prec: i64) -> Option<Arb> {
        let mut out = Arb::new();
        let n = &args[0];
        let x = args.last().unwrap();
        unsafe {
            match family {
                OrthoFamily::Legendre => {
                    let zero = Arb::from_i64(0);
                    ffi::arb_hypgeom_legendre_p(
                        out.as_mut_ptr(),
                        n.as_ptr(),
                        zero.as_ptr(),
                        x.as_ptr(),
                        0,
                        prec,
                    )
                }
                OrthoFamily::AssocLegendre => ffi::arb_hypgeom_legendre_p(
                    out.as_mut_ptr(),
                    n.as_ptr(),
                    args[1].as_ptr(),
                    x.as_ptr(),
                    0,
                    prec,
                ),
                OrthoFamily::HermiteH => {
                    ffi::arb_hypgeom_hermite_h(out.as_mut_ptr(), n.as_ptr(), x.as_ptr(), prec)
                }
                OrthoFamily::HermiteHe => {
                    // He_ν(x) = 2^{−ν/2} H_ν(x/√2)
                    let two = Arb::from_i64(2);
                    let z = x.div(&two.sqrt(prec), prec);
                    let mut h = Arb::new();
                    ffi::arb_hypgeom_hermite_h(h.as_mut_ptr(), n.as_ptr(), z.as_ptr(), prec);
                    let e = n.neg().mul_2exp(-1);
                    out = h.mul(&two.pow(&e, prec), prec);
                }
                OrthoFamily::Laguerre => {
                    let zero = Arb::from_i64(0);
                    ffi::arb_hypgeom_laguerre_l(
                        out.as_mut_ptr(),
                        n.as_ptr(),
                        zero.as_ptr(),
                        x.as_ptr(),
                        prec,
                    )
                }
                OrthoFamily::AssocLaguerre => ffi::arb_hypgeom_laguerre_l(
                    out.as_mut_ptr(),
                    n.as_ptr(),
                    args[1].as_ptr(),
                    x.as_ptr(),
                    prec,
                ),
                OrthoFamily::ChebyshevT => {
                    ffi::arb_hypgeom_chebyshev_t(out.as_mut_ptr(), n.as_ptr(), x.as_ptr(), prec)
                }
                OrthoFamily::ChebyshevU => {
                    ffi::arb_hypgeom_chebyshev_u(out.as_mut_ptr(), n.as_ptr(), x.as_ptr(), prec)
                }
                OrthoFamily::Gegenbauer => ffi::arb_hypgeom_gegenbauer_c(
                    out.as_mut_ptr(),
                    n.as_ptr(),
                    args[1].as_ptr(),
                    x.as_ptr(),
                    prec,
                ),
                OrthoFamily::Jacobi => ffi::arb_hypgeom_jacobi_p(
                    out.as_mut_ptr(),
                    n.as_ptr(),
                    args[1].as_ptr(),
                    args[2].as_ptr(),
                    x.as_ptr(),
                    prec,
                ),
            }
        }
        out.is_finite().then_some(out)
    }

    pub(super) fn f64_value(family: OrthoFamily, args: &[f64]) -> Option<f64> {
        if !available() {
            return None;
        }
        let balls: Vec<Arb> = args.iter().map(|&v| Arb::from_f64(v)).collect();
        for prec in [128_i64, 256, 512] {
            if let Some(v) = call(family, &balls, prec) {
                if v.rel_accuracy_bits() >= 53 {
                    let m = v.midpoint_f64();
                    return m.is_finite().then_some(m);
                }
                // An exact zero has no relative accuracy; accept a tight one.
                if v.contains_zero() && v.radius_f64() < 1e-300 {
                    return Some(0.0);
                }
            }
        }
        None
    }

    pub(super) fn ball_value(family: OrthoFamily, args: &[ArbBall]) -> Option<ArbBall> {
        if !available() {
            return None;
        }
        let prec = args.iter().map(|b| b.prec).max().unwrap_or(128);
        let mut arbs = Vec::with_capacity(args.len());
        for b in args {
            if b.mid.is_nan() || b.rad.is_nan() || b.mid.is_infinite() {
                return None;
            }
            arbs.push(Arb::from_mid_rad(&b.mid, &b.rad)?);
        }
        let v = call(family, &arbs, i64::from(prec) + 16)?;
        let mid = v.midpoint_rug()?;
        let rad = v.radius_rug()?;
        // Round the midpoint to `prec` bits and fold the rounding error into
        // the radius, rounding the radius up.
        let mid_r = Float::with_val(prec, &mid);
        let err = Float::with_val(mid.prec().max(prec) + 2, &mid - &mid_r).abs();
        let rad_r = Float::with_val_round(prec, &rad + &err, Round::Up).0;
        Some(ArbBall {
            mid: mid_r,
            rad: rad_r,
            prec,
        })
    }
}

#[cfg(flint_arb)]
fn arb_f64(family: OrthoFamily, args: &[f64]) -> Option<f64> {
    arb_impl::f64_value(family, args)
}

#[cfg(not(flint_arb))]
fn arb_f64(_family: OrthoFamily, _args: &[f64]) -> Option<f64> {
    None
}

/// Rigorous enclosure of `family(args…)`; `None` without Arb, for an odd
/// Legendre order off `[−1, 1]`, or when Arb cannot produce a finite ball.
pub(crate) fn eval_ball(family: OrthoFamily, args: &[ArbBall]) -> Option<ArbBall> {
    if args.len() != family.arity() {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_impl::ball_value(family, args)
    }
    #[cfg(not(flint_arb))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrences_match_closed_forms() {
        let x: f64 = 0.37;
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-14 * (1.0 + b.abs());
        let p3 = 0.5 * (5.0 * x.powi(3) - 3.0 * x);
        assert!(close(
            eval_f64(OrthoFamily::Legendre, &[3.0, x]).unwrap(),
            p3
        ));
        assert!(close(
            eval_f64(OrthoFamily::Legendre, &[-4.0, x]).unwrap(),
            p3
        ));
        let h3 = 8.0 * x.powi(3) - 12.0 * x;
        assert!(close(
            eval_f64(OrthoFamily::HermiteH, &[3.0, x]).unwrap(),
            h3
        ));
        // P_2^1(x) = −3x√(1−x²); P_2^{−1} = (1/6)·3x√(1−x²)
        let s = (1.0 - x * x).sqrt();
        assert!(close(
            eval_f64(OrthoFamily::AssocLegendre, &[2.0, 1.0, x]).unwrap(),
            -3.0 * x * s
        ));
        assert!(close(
            eval_f64(OrthoFamily::AssocLegendre, &[2.0, -1.0, x]).unwrap(),
            0.5 * x * s
        ));
        assert_eq!(
            eval_f64(OrthoFamily::AssocLegendre, &[2.0, 1.0, 1.5]),
            None,
            "odd order off [−1, 1] is imaginary"
        );
        assert_eq!(eval_f64(OrthoFamily::ChebyshevU, &[-1.0, x]), Some(0.0));
        // U_{−3} = −U_1 = −2x
        assert!(close(
            eval_f64(OrthoFamily::ChebyshevU, &[-3.0, x]).unwrap(),
            -2.0 * x
        ));
    }

    #[test]
    fn jacobi_recurrence_and_sum_agree() {
        for &(a, b) in &[(0.5, -0.25), (2.0, 3.0), (-0.5, -0.5)] {
            for n in 0..12 {
                let r = jacobi(n, a, b, 0.3);
                let s = jacobi_sum(n, a, b, 0.3);
                assert!(
                    (r - s).abs() < 1e-13 * (1.0 + s.abs()),
                    "{n} {a} {b}: {r} {s}"
                );
            }
        }
        // a + b = −2 makes the n = 2 recurrence denominator vanish (k+a+b = 0).
        let v = jacobi(2, -1.0, -1.0, 0.3);
        assert!((v - jacobi_sum(2, -1.0, -1.0, 0.3)).abs() < 1e-14);
    }

    #[test]
    fn hermite_overflow_is_none() {
        assert_eq!(eval_f64(OrthoFamily::HermiteH, &[400.0, 30.0]), None);
    }
}
