//! Numerical kernels for the Bessel, Airy, zeta and polylogarithm primitives.
//!
//! Every value — `f64`, rigorous ball and complex `f64` — comes from FLINT's
//! Arb layer (`arb_hypgeom_bessel_*`, `arb_hypgeom_airy`, `arb_zeta`,
//! `arb_hurwitz_zeta`, `arb_polylog` and their `acb_*` counterparts) when the
//! build carries it (`cfg(flint_arb)`); without it every kernel declines and
//! the capability probe reports no numeric slot.
//!
//! * **`f64`** is accepted only once Arb certifies 53 correct bits, retrying
//!   at 128, 256 and 512 bits of working precision, so a value next to a zero
//!   of `J_ν` is still correctly rounded to a few ulps rather than whatever a
//!   fixed-precision series leaves.
//! * **Balls** are evaluated 16 bits above the inputs' precision and the
//!   midpoint rounding is folded into the radius.
//! * A value that is not real (`Y_ν`, `K_ν` at `x < 0`, `J_ν` at `x < 0` for
//!   non-integer `ν`, `Li_s(x)` for `x > 1`) and a pole (`ζ(1)`) come back as
//!   `None` — Arb returns an indeterminate ball there and this module never
//!   substitutes a real part.

use crate::ball::ArbBall;

/// One special-function kernel; the argument layout is its primitive's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    BesselJ,
    BesselY,
    BesselI,
    BesselK,
    SphericalJ,
    SphericalY,
    AiryAi,
    AiryBi,
    AiryAiPrime,
    AiryBiPrime,
    Zeta,
    HurwitzZeta,
    Polylog,
}

impl Kernel {
    pub(crate) const ALL: [Kernel; 13] = [
        Kernel::BesselJ,
        Kernel::BesselY,
        Kernel::BesselI,
        Kernel::BesselK,
        Kernel::SphericalJ,
        Kernel::SphericalY,
        Kernel::AiryAi,
        Kernel::AiryBi,
        Kernel::AiryAiPrime,
        Kernel::AiryBiPrime,
        Kernel::Zeta,
        Kernel::HurwitzZeta,
        Kernel::Polylog,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Kernel::BesselJ => "bessel_j",
            Kernel::BesselY => "bessel_y",
            Kernel::BesselI => "bessel_i",
            Kernel::BesselK => "bessel_k",
            Kernel::SphericalJ => "spherical_jn",
            Kernel::SphericalY => "spherical_yn",
            Kernel::AiryAi => "airy_ai",
            Kernel::AiryBi => "airy_bi",
            Kernel::AiryAiPrime => "airy_ai_prime",
            Kernel::AiryBiPrime => "airy_bi_prime",
            Kernel::Zeta => "zeta",
            Kernel::HurwitzZeta => "hurwitz_zeta",
            Kernel::Polylog => "polylog",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Kernel> {
        Kernel::ALL.into_iter().find(|k| k.name() == name)
    }

    pub(crate) fn arity(self) -> usize {
        match self {
            Kernel::AiryAi
            | Kernel::AiryBi
            | Kernel::AiryAiPrime
            | Kernel::AiryBiPrime
            | Kernel::Zeta => 1,
            _ => 2,
        }
    }
}

/// `f64` value of `kernel(args…)`, or `None` (no Arb, out of the real
/// domain, a pole, or no certified value).
pub(crate) fn eval_f64(kernel: Kernel, args: &[f64]) -> Option<f64> {
    if args.len() != kernel.arity() || args.iter().any(|a| !a.is_finite()) {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_impl::f64_value(kernel, args)
    }
    #[cfg(not(flint_arb))]
    {
        None
    }
}

/// Rigorous enclosure of `kernel(args…)`.
pub(crate) fn eval_ball(kernel: Kernel, args: &[ArbBall]) -> Option<ArbBall> {
    if args.len() != kernel.arity() {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_impl::ball_value(kernel, args)
    }
    #[cfg(not(flint_arb))]
    {
        None
    }
}

/// Complex `f64` value of `kernel(args…)` (principal branches, as Arb and
/// DLMF define them), certified to 53 bits; `None` otherwise.
pub(crate) fn eval_c64(kernel: Kernel, args: &[(f64, f64)]) -> Option<(f64, f64)> {
    if args.len() != kernel.arity()
        || args
            .iter()
            .any(|(re, im)| !re.is_finite() || !im.is_finite())
    {
        return None;
    }
    #[cfg(flint_arb)]
    {
        arb_impl::c64_value(kernel, args)
    }
    #[cfg(not(flint_arb))]
    {
        None
    }
}

#[cfg(flint_arb)]
mod arb_impl {
    use super::Kernel;
    use crate::ball::ArbBall;
    use crate::flint::acb::{Acb, Arb};
    use crate::flint::arb as ffi;
    use rug::float::Round;
    use rug::Float;

    fn available() -> bool {
        ffi::abi_self_check().is_ok()
    }

    /// The order ball as an integer, when it is an exact integer.
    fn exact_int(b: &Arb) -> Option<i64> {
        if !b.is_exact() {
            return None;
        }
        let v = b.midpoint_f64();
        (v.is_finite() && v == v.trunc() && v.abs() < 1e15).then_some(v as i64)
    }

    fn is_negative(b: &Arb) -> bool {
        b.neg().is_positive()
    }

    fn bessel(kernel: Kernel, nu: &Arb, x: &Arb, prec: i64) -> Arb {
        let mut out = Arb::new();
        unsafe {
            let f = match kernel {
                Kernel::BesselJ => ffi::arb_hypgeom_bessel_j,
                Kernel::BesselY => ffi::arb_hypgeom_bessel_y,
                Kernel::BesselI => ffi::arb_hypgeom_bessel_i,
                _ => ffi::arb_hypgeom_bessel_k,
            };
            f(out.as_mut_ptr(), nu.as_ptr(), x.as_ptr(), prec);
        }
        out
    }

    /// `j_n(x) = √(π/(2x)) J_{n+½}(x)` (DLMF 10.47.3), `y_n` likewise, for an
    /// exact integer `n`; `x < 0` by parity (`j_n(−x) = (−1)ⁿ j_n(x)`,
    /// `y_n(−x) = (−1)^{n+1} y_n(x)`, DLMF 10.47.14).
    fn spherical(kernel: Kernel, n: &Arb, x: &Arb, prec: i64) -> Option<Arb> {
        let ni = exact_int(n)?;
        if x.is_exact() && x.contains_zero() {
            // j_0(0) = 1, j_n(0) = 0 for n ≥ 1; y_n and j_{n<0} are singular.
            return match (kernel, ni) {
                (Kernel::SphericalJ, 0) => Some(Arb::from_i64(1)),
                (Kernel::SphericalJ, k) if k > 0 => Some(Arb::from_i64(0)),
                _ => None,
            };
        }
        let (ax, flip) = if x.is_positive() {
            (x.clone(), false)
        } else if is_negative(x) {
            (x.neg(), true)
        } else {
            return None; // straddles 0
        };
        let half = Arb::from_i64(1).mul_2exp(-1);
        let nu = n.add(&half, prec);
        let cyl = if kernel == Kernel::SphericalJ {
            Kernel::BesselJ
        } else {
            Kernel::BesselY
        };
        let jb = bessel(cyl, &nu, &ax, prec);
        // √(π/2) / √x
        let scale = Arb::pi(prec)
            .mul_2exp(-1)
            .sqrt(prec)
            .div(&ax.sqrt(prec), prec);
        let mut v = jb.mul(&scale, prec);
        if flip {
            let odd = match kernel {
                Kernel::SphericalJ => ni.rem_euclid(2) == 1,
                _ => ni.rem_euclid(2) == 0,
            };
            if odd {
                v = v.neg();
            }
        }
        Some(v)
    }

    /// `kernel(args)` over Arb balls at `prec` bits.
    fn call(kernel: Kernel, args: &[Arb], prec: i64) -> Option<Arb> {
        let mut out = Arb::new();
        unsafe {
            match kernel {
                Kernel::BesselJ | Kernel::BesselY | Kernel::BesselI | Kernel::BesselK => {
                    out = bessel(kernel, &args[0], &args[1], prec);
                }
                Kernel::SphericalJ | Kernel::SphericalY => {
                    out = spherical(kernel, &args[0], &args[1], prec)?;
                }
                Kernel::AiryAi => ffi::arb_hypgeom_airy(
                    out.as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryAiPrime => ffi::arb_hypgeom_airy(
                    std::ptr::null_mut(),
                    out.as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryBi => ffi::arb_hypgeom_airy(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    out.as_mut_ptr(),
                    std::ptr::null_mut(),
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryBiPrime => ffi::arb_hypgeom_airy(
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::Zeta => ffi::arb_zeta(out.as_mut_ptr(), args[0].as_ptr(), prec),
                Kernel::HurwitzZeta => ffi::arb_hurwitz_zeta(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::Polylog => {
                    ffi::arb_polylog(out.as_mut_ptr(), args[0].as_ptr(), args[1].as_ptr(), prec)
                }
            }
        }
        out.is_finite().then_some(out)
    }

    pub(super) fn f64_value(kernel: Kernel, args: &[f64]) -> Option<f64> {
        if !available() {
            return None;
        }
        let balls: Vec<Arb> = args.iter().map(|&v| Arb::from_f64(v)).collect();
        for prec in [128_i64, 256, 512] {
            if let Some(v) = call(kernel, &balls, prec) {
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

    pub(super) fn ball_value(kernel: Kernel, args: &[ArbBall]) -> Option<ArbBall> {
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
        let v = call(kernel, &arbs, i64::from(prec) + 16)?;
        let mid = v.midpoint_rug()?;
        let rad = v.radius_rug()?;
        let mid_r = Float::with_val(prec, &mid);
        let err = Float::with_val(mid.prec().max(prec) + 2, &mid - &mid_r).abs();
        let rad_r = Float::with_val_round(prec, &rad + &err, Round::Up).0;
        Some(ArbBall {
            mid: mid_r,
            rad: rad_r,
            prec,
        })
    }

    fn call_c(kernel: Kernel, args: &[Acb], prec: i64) -> Option<Acb> {
        let mut out = Acb::new();
        let null = std::ptr::null_mut();
        unsafe {
            match kernel {
                Kernel::BesselJ => ffi::acb_hypgeom_bessel_j(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::BesselY => ffi::acb_hypgeom_bessel_y(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::BesselI => ffi::acb_hypgeom_bessel_i(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::BesselK => ffi::acb_hypgeom_bessel_k(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::SphericalJ | Kernel::SphericalY => {
                    // √(π/2)·F_{n+½}(z)/√z — the product is entire in z for
                    // integer n, and both factors use the principal branch.
                    let half = Acb::from_f64(0.5, 0.0);
                    let nu = args[0].add(&half, prec);
                    let mut f = Acb::new();
                    let g = if kernel == Kernel::SphericalJ {
                        ffi::acb_hypgeom_bessel_j
                    } else {
                        ffi::acb_hypgeom_bessel_y
                    };
                    g(f.as_mut_ptr(), nu.as_ptr(), args[1].as_ptr(), prec);
                    let scale = Acb::pi(prec).mul_2exp(-1).sqrt(prec);
                    out = f.mul(&scale, prec).div(&args[1].sqrt(prec), prec);
                }
                Kernel::AiryAi => ffi::acb_hypgeom_airy(
                    out.as_mut_ptr(),
                    null,
                    null,
                    null,
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryAiPrime => ffi::acb_hypgeom_airy(
                    null,
                    out.as_mut_ptr(),
                    null,
                    null,
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryBi => ffi::acb_hypgeom_airy(
                    null,
                    null,
                    out.as_mut_ptr(),
                    null,
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::AiryBiPrime => ffi::acb_hypgeom_airy(
                    null,
                    null,
                    null,
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    prec,
                ),
                Kernel::Zeta => ffi::acb_zeta(out.as_mut_ptr(), args[0].as_ptr(), prec),
                Kernel::HurwitzZeta => ffi::acb_hurwitz_zeta(
                    out.as_mut_ptr(),
                    args[0].as_ptr(),
                    args[1].as_ptr(),
                    prec,
                ),
                Kernel::Polylog => {
                    ffi::acb_polylog(out.as_mut_ptr(), args[0].as_ptr(), args[1].as_ptr(), prec)
                }
            }
        }
        out.is_finite().then_some(out)
    }

    fn part_ok(p: &Arb) -> Option<f64> {
        if p.rel_accuracy_bits() >= 53 {
            let m = p.midpoint_f64();
            return m.is_finite().then_some(m);
        }
        (p.contains_zero() && p.radius_f64() < 1e-300).then_some(0.0)
    }

    pub(super) fn c64_value(kernel: Kernel, args: &[(f64, f64)]) -> Option<(f64, f64)> {
        if !available() {
            return None;
        }
        let balls: Vec<Acb> = args.iter().map(|&(r, i)| Acb::from_f64(r, i)).collect();
        for prec in [128_i64, 256, 512] {
            if let Some(v) = call_c(kernel, &balls, prec) {
                // Each component to 53 bits, or a component that is tightly
                // zero (a real value has an exactly-zero imaginary part).
                if let (Some(re), Some(im)) = (part_ok(&v.real()), part_ok(&v.imag())) {
                    return Some((re, im));
                }
                // A tiny component next to a large one: accept when the
                // whole value carries 53 bits relative to its modulus.
                if v.rel_accuracy_bits() >= 53 {
                    let (re, im) = (v.real().midpoint_f64(), v.imag().midpoint_f64());
                    if re.is_finite() && im.is_finite() {
                        return Some((re, im));
                    }
                }
            }
        }
        None
    }
}
