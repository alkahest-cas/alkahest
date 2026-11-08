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
    /// `ζ′(s)` — the first derivative of the Riemann zeta function.
    ZetaPrime,
    /// `H^{(1)}_ν(x) = J_ν(x) + i Y_ν(x)` — complex-valued, so it has a
    /// complex kernel only.
    HankelH1,
    /// `H^{(2)}_ν(x) = J_ν(x) − i Y_ν(x)`.
    HankelH2,
    /// `log Γ(x)` (`loggamma`): real for `x > 0`; the principal branch (the
    /// analytic continuation from the positive axis, Arb's `acb_lgamma`)
    /// for complex arguments.
    LogGamma,
    /// `ζ⁽ⁿ⁾(s)` (`zeta_deriv(n, s)`), the `n`-th derivative of the Riemann
    /// zeta function for an exact integer `n ≥ 0`, from Arb's zeta jet.
    ZetaDeriv,
}

impl Kernel {
    pub(crate) const ALL: [Kernel; 18] = [
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
        Kernel::ZetaPrime,
        Kernel::HankelH1,
        Kernel::HankelH2,
        Kernel::LogGamma,
        Kernel::ZetaDeriv,
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
            Kernel::ZetaPrime => "zeta_prime",
            Kernel::HankelH1 => "hankel1",
            Kernel::HankelH2 => "hankel2",
            Kernel::LogGamma => "loggamma",
            Kernel::ZetaDeriv => "zeta_deriv",
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
            | Kernel::Zeta
            | Kernel::ZetaPrime
            | Kernel::LogGamma => 1,
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

/// `ζ⁽ᵏ⁾(s)/k!` for `k < len` at a real ball `s`, valid for every point of the
/// ball; `None` when the ball contains the pole `s = 1`, or without Arb.
pub(crate) fn zeta_jet_ball(s: &ArbBall, len: usize) -> Option<Vec<ArbBall>> {
    #[cfg(flint_arb)]
    {
        arb_impl::zeta_jet_ball(s, len)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (s, len);
        None
    }
}

/// A rigorous enclosure of the Stieltjes constant `γₙ` (`n ≥ 1`) at `prec`
/// bits, from the deflated zeta jet at `s = 1` (`ζ(1+x) − 1/x = Σ (−1)ⁿ γₙ xⁿ/n!`);
/// `None` without Arb.
pub(crate) fn stieltjes_ball(n: u32, prec: u32) -> Option<ArbBall> {
    #[cfg(flint_arb)]
    {
        arb_impl::stieltjes_ball(n, prec)
    }
    #[cfg(not(flint_arb))]
    {
        let _ = (n, prec);
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
pub(super) mod arb_impl {
    use super::Kernel;
    use crate::ball::ArbBall;
    use crate::flint::acb::{Acb, AcbVec, Arb};
    use crate::flint::arb as ffi;
    use rug::float::Round;
    use rug::Float;

    pub(in crate::specfun) fn available() -> bool {
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
                Kernel::ZetaPrime => {
                    let s = Acb::from_parts(&args[0], &Arb::from_i64(0));
                    let jet = zeta_jet_acb(&s, 2, prec)?;
                    // Real `s` gives a real derivative; the imaginary part is
                    // an enclosure of 0.
                    if !jet[1].imag().contains_zero() {
                        return None;
                    }
                    out = jet[1].real();
                }
                Kernel::ZetaDeriv => {
                    let n = zeta_deriv_order(&args[0])?;
                    let s = Acb::from_parts(&args[1], &Arb::from_i64(0));
                    let v = zeta_deriv_acb(n, &s, prec)?;
                    if !v.imag().contains_zero() {
                        return None;
                    }
                    out = v.real();
                }
                // Complex-valued: no real kernel.
                Kernel::HankelH1 | Kernel::HankelH2 => return None,
                // Real only on the positive axis: for x < 0 the principal
                // branch is complex (Im = −kπ between the poles), so the real
                // evaluator declines instead of returning log|Γ(x)|.
                Kernel::LogGamma => {
                    if !args[0].is_positive() {
                        return None;
                    }
                    ffi::arb_lgamma(out.as_mut_ptr(), args[0].as_ptr(), prec)
                }
            }
        }
        if kernel == Kernel::Polylog && !out.is_finite() {
            // The real `arb_polylog` answers on the positive axis only and is
            // indeterminate for z < 0; the complex routine has it, and the
            // value is real exactly when its imaginary part encloses 0 (for
            // z > 1 it does not, and the result stays declined).
            let s = Acb::from_parts(&args[0], &Arb::from_i64(0));
            let z = Acb::from_parts(&args[1], &Arb::from_i64(0));
            let mut w = Acb::new();
            unsafe { ffi::acb_polylog(w.as_mut_ptr(), s.as_ptr(), z.as_ptr(), prec) };
            if w.is_finite() && w.imag().contains_zero() {
                out = w.real();
            }
        }
        out.is_finite().then_some(out)
    }

    /// The first `len` Taylor coefficients `ζ⁽ᵏ⁾(s)/k!` at a complex `s`
    /// (`acb_dirichlet_zeta_jet`, no deflation, so the pole at `s = 1` is an
    /// indeterminate ball and comes back as `None`).
    fn zeta_jet_acb(s: &Acb, len: usize, prec: i64) -> Option<Vec<Acb>> {
        let one = Acb::from_f64(1.0, 0.0);
        if s.contains(&one) {
            return None;
        }
        let mut v = AcbVec::new(len);
        unsafe {
            ffi::acb_dirichlet_zeta_jet(v.as_mut_ptr(), s.as_ptr(), 0, len as i64, prec);
        }
        let out: Vec<Acb> = (0..len).map(|i| v.get(i)).collect();
        out.iter().all(Acb::is_finite).then_some(out)
    }

    /// Largest derivative order `zeta_deriv` evaluates (the jet is `O(n²)`).
    const MAX_ZETA_DERIV: i64 = 200;

    /// The order of `zeta_deriv(n, s)`: an exact integer in `[0, 200]`.
    fn zeta_deriv_order(b: &Arb) -> Option<usize> {
        let n = exact_int(b)?;
        (0..=MAX_ZETA_DERIV).contains(&n).then_some(n as usize)
    }

    /// `ζ⁽ᵏ⁾(s)/k!` for `k < len` and `Re s < 0`, from the functional equation
    /// `ζ(u) = 2^u π^{u−1} sin(πu/2) Γ(1−u) ζ(1−u)` expanded at `u = s + t`.
    ///
    /// Arb's own jet reflects too, but past the second coefficient its radius
    /// stops improving near `10⁻¹⁷` whatever the working precision, so it can
    /// never certify 53 bits there. `ζ(1−s−t)` is in the half-plane where the
    /// jet converges at full precision.
    fn zeta_jet_reflected(s: &Acb, len: usize, prec: i64) -> Option<Vec<Acb>> {
        let one = Acb::from_f64(1.0, 0.0);
        let two = Acb::from_f64(2.0, 0.0);
        let pi = Acb::pi(prec);
        let s1 = one.sub(s, prec);
        // (−1)^k e_k: the coefficients of ζ(1−s−t) in t.
        let zeta: Vec<Acb> = zeta_jet_acb(&s1, len, prec)?
            .into_iter()
            .enumerate()
            .map(|(k, c)| if k % 2 == 1 { c.neg() } else { c })
            .collect();
        let mut h = AcbVec::new(2);
        h.set(0, &s1);
        h.set(1, &one.neg());
        let mut g = AcbVec::new(len);
        unsafe {
            ffi::_acb_poly_gamma_series(g.as_mut_ptr(), h.as_ptr(), 2, len as i64, prec);
        }
        let gamma: Vec<Acb> = (0..len).map(|i| g.get(i)).collect();
        // (2π)^t: L^k/k!; sin(θ + (π/2)t): (π/2)^k/k! · sin(θ + kπ/2).
        let l = two.mul(&pi, prec).log(prec);
        let half_pi = pi.mul_2exp(-1);
        let theta = half_pi.mul(s, prec);
        let i = Acb::i();
        let (e_p, e_m) = (
            i.mul(&theta, prec).exp(prec),
            i.neg().mul(&theta, prec).exp(prec),
        );
        let sin = e_p.sub(&e_m, prec).div(&two.mul(&i, prec), prec);
        let cos = e_p.add(&e_m, prec).mul_2exp(-1);
        let (neg_sin, neg_cos) = (sin.neg(), cos.neg());
        let (mut pow_l, mut pow_p) = (one.clone(), one.clone());
        let mut power = Vec::with_capacity(len);
        let mut sine = Vec::with_capacity(len);
        for k in 0..len {
            power.push(pow_l.clone());
            sine.push(pow_p.mul(
                match k % 4 {
                    0 => &sin,
                    1 => &cos,
                    2 => &neg_sin,
                    _ => &neg_cos,
                },
                prec,
            ));
            let kp1 = Acb::from_f64((k + 1) as f64, 0.0);
            pow_l = pow_l.mul(&l, prec).div(&kp1, prec);
            pow_p = pow_p.mul(&half_pi, prec).div(&kp1, prec);
        }
        let mul = |a: &[Acb], b: &[Acb]| -> Vec<Acb> {
            (0..len)
                .map(|m| {
                    let mut t = Acb::new();
                    for j in 0..=m {
                        t = t.add(&a[j].mul(&b[m - j], prec), prec);
                    }
                    t
                })
                .collect()
        };
        let prefactor = two
            .pow(s, prec)
            .mul(&pi.pow(&s.sub(&one, prec), prec), prec);
        let product = mul(&mul(&mul(&power, &sine), &gamma), &zeta);
        let out: Vec<Acb> = product.iter().map(|c| c.mul(&prefactor, prec)).collect();
        out.iter().all(Acb::is_finite).then_some(out)
    }

    /// `ζ⁽ⁿ⁾(s) = n! · [tⁿ] ζ(s + t)` from the zeta jet.
    fn zeta_deriv_acb(n: usize, s: &Acb, prec: i64) -> Option<Acb> {
        let wp = prec + 2 * n as i64;
        let c = if s.real().neg().is_positive() {
            zeta_jet_reflected(s, n + 1, wp)?.swap_remove(n)
        } else {
            zeta_jet_acb(s, n + 1, wp)?.swap_remove(n)
        };
        let mut fact = Arb::from_i64(1);
        for k in 2..=n as i64 {
            fact = fact.mul(&Arb::from_i64(k), wp);
        }
        let v = c.mul(&Acb::from_parts(&fact, &Arb::from_i64(0)), wp);
        v.is_finite().then_some(v)
    }

    /// `γₙ = (−1)ⁿ n! · [xⁿ](ζ(1+x) − 1/x)`.
    pub(in crate::specfun) fn stieltjes_ball(n: u32, prec: u32) -> Option<ArbBall> {
        if !available() || n > 200 {
            return None;
        }
        let wp = i64::from(prec) + 32 + 4 * i64::from(n);
        let one = Acb::from_f64(1.0, 0.0);
        let len = n as usize + 1;
        let mut v = AcbVec::new(len);
        unsafe {
            ffi::acb_dirichlet_zeta_jet(v.as_mut_ptr(), one.as_ptr(), 1, len as i64, wp);
        }
        let c = v.get(n as usize);
        if !c.is_finite() || !c.imag().contains_zero() {
            return None;
        }
        let t = from_arb(&c.real(), prec + 32)?;
        let mut f = rug::Integer::from(1);
        for k in 2..=n {
            f *= k;
        }
        if n % 2 == 1 {
            f = -f;
        }
        let scaled = t.mul_ref(&ArbBall::from_integer(&f, prec + 32));
        let mid = Float::with_val(prec, &scaled.mid);
        let err = Float::with_val(scaled.mid.prec() + 2, &scaled.mid - &mid).abs();
        let rad = Float::with_val_round(prec, &scaled.rad + &err, Round::Up).0;
        Some(ArbBall { mid, rad, prec })
    }

    /// `ζ⁽ᵏ⁾(s)/k!`, `k < len`, as rigorous real balls valid for **every**
    /// point of the real ball `s`.
    pub(in crate::specfun) fn zeta_jet_ball(s: &ArbBall, len: usize) -> Option<Vec<ArbBall>> {
        if !available() {
            return None;
        }
        let sb = to_arb(s)?;
        let sc = Acb::from_parts(&sb, &Arb::from_i64(0));
        let jet = zeta_jet_acb(&sc, len, i64::from(s.prec) + 16)?;
        jet.iter()
            .map(|c| {
                if !c.imag().contains_zero() {
                    return None;
                }
                from_arb(&c.real(), s.prec)
            })
            .collect()
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
                Kernel::ZetaPrime => {
                    out = zeta_jet_acb(&args[0], 2, prec)?.swap_remove(1);
                }
                Kernel::ZetaDeriv => {
                    // The order must be a real exact integer.
                    if !args[0].imag().is_exact() || !args[0].imag().contains_zero() {
                        return None;
                    }
                    let n = zeta_deriv_order(&args[0].real())?;
                    out = zeta_deriv_acb(n, &args[1], prec)?;
                }
                Kernel::LogGamma => ffi::acb_lgamma(out.as_mut_ptr(), args[0].as_ptr(), prec),
                Kernel::HankelH1 | Kernel::HankelH2 => {
                    let mut j = Acb::new();
                    let mut y = Acb::new();
                    ffi::acb_hypgeom_bessel_j(
                        j.as_mut_ptr(),
                        args[0].as_ptr(),
                        args[1].as_ptr(),
                        prec,
                    );
                    ffi::acb_hypgeom_bessel_y(
                        y.as_mut_ptr(),
                        args[0].as_ptr(),
                        args[1].as_ptr(),
                        prec,
                    );
                    let iy = y.mul(&Acb::i(), prec);
                    out = if kernel == Kernel::HankelH1 {
                        j.add(&iy, prec)
                    } else {
                        j.sub(&iy, prec)
                    };
                }
            }
        }
        out.is_finite().then_some(out)
    }

    /// An [`ArbBall`] as an Arb ball (exactly), or `None` for a non-finite one.
    pub(in crate::specfun) fn to_arb(b: &ArbBall) -> Option<Arb> {
        if b.mid.is_nan() || b.rad.is_nan() || b.mid.is_infinite() {
            return None;
        }
        Arb::from_mid_rad(&b.mid, &b.rad)
    }

    /// An Arb ball as an [`ArbBall`] at `prec` bits, the midpoint rounding
    /// folded into the radius.
    pub(in crate::specfun) fn from_arb(v: &Arb, prec: u32) -> Option<ArbBall> {
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

    /// The midpoint of `v` when it carries 53 correct bits (or is a tight
    /// zero).
    pub(in crate::specfun) fn certified_f64(v: &Arb) -> Option<f64> {
        if !v.is_finite() {
            return None;
        }
        part_ok(v)
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
            if let Some(v) = call_c(kernel, &balls, prec)
                .as_ref()
                .and_then(certified_c64)
            {
                return Some(v);
            }
        }
        None
    }

    /// The midpoint of a complex ball when it carries 53 correct bits: each
    /// component to 53 bits (or tightly zero — a real value has an exactly
    /// zero imaginary part), or the whole value to 53 bits relative to its
    /// modulus (a tiny component next to a large one).
    pub(in crate::specfun) fn certified_c64(v: &Acb) -> Option<(f64, f64)> {
        if !v.is_finite() {
            return None;
        }
        if let (Some(re), Some(im)) = (part_ok(&v.real()), part_ok(&v.imag())) {
            return Some((re, im));
        }
        if v.rel_accuracy_bits() >= 53 {
            let (re, im) = (v.real().midpoint_f64(), v.imag().midpoint_f64());
            if re.is_finite() && im.is_finite() {
                return Some((re, im));
            }
        }
        None
    }
}
