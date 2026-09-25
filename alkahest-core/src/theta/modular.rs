//! Genus-1 modular and elliptic functions: `eta`, `j`, `lambda`, `Delta`, the
//! Eisenstein series, the four Jacobi theta functions, and the Weierstrass
//! family, each as a rigorous enclosure.
//!
//! Every function here needs `tau` in the upper half-plane and says so: if
//! `Im(tau) > 0` cannot be *proved* at the working precision the call refuses
//! with [`ThetaError::NotInUpperHalfPlane`] rather than handing back the
//! infinite-radius ball FLINT would otherwise produce. The distinction matters
//! because "I could not tell" and "the answer is unbounded" are different
//! facts, and only one of them is fixed by asking again with more precision.
//!
//! The `z` argument of [`jacobi_theta`] and the Weierstrass functions is *not*
//! constrained: those are entire (or meromorphic) in `z`, and a `z` at a
//! lattice point simply comes back indeterminate, which is the honest answer
//! for a pole.

use super::ball::ComplexBall;
use super::error::ThetaError;
use super::{backend, evaluate_at, EllipticFn, ModularFn, Precision};

/// Refuse unless `Im(tau)` is certainly positive at this working precision.
fn require_upper_half_plane(function: &'static str, tau: &ComplexBall) -> Result<(), ThetaError> {
    if backend::imag_is_certainly_positive(tau)? {
        Ok(())
    } else {
        Err(ThetaError::NotInUpperHalfPlane { function })
    }
}

fn modular(
    function: &'static str,
    which: ModularFn,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    require_upper_half_plane(function, tau)?;
    evaluate_at(function, precision, |bits| {
        backend::modular_scalar(which, tau, bits)
    })
}

/// The Dedekind eta function `eta(tau) = q^{1/24} prod_{n>=1} (1 - q^n)`,
/// `q = exp(2 pi i tau)`.
///
/// `eta(i) = Gamma(1/4) / (2 pi^{3/4}) = 0.76822542...`, which is the
/// normalisation check this wrapper is pinned against.
pub fn dedekind_eta(tau: &ComplexBall, precision: Precision) -> Result<ComplexBall, ThetaError> {
    modular("eta", ModularFn::Eta, tau, precision)
}

/// Klein's `j`-invariant, normalised so that `j(i) = 1728` and `j(rho) = 0` for
/// `rho = exp(2 pi i / 3)`.
///
/// Those two values are the classical normalisation checkpoints and both are
/// asserted in this module's tests. Note that `j(rho) = 0` can only be checked
/// with [`Precision::Bits`] plus [`ComplexBall::contains_zero`]: a value that is
/// exactly zero has no relative accuracy for [`Precision::AccurateTo`] to
/// reach, and asking for it will correctly refuse.
pub fn j_invariant(tau: &ComplexBall, precision: Precision) -> Result<ComplexBall, ThetaError> {
    modular("j", ModularFn::JInvariant, tau, precision)
}

/// The modular lambda function `lambda(tau)`, the elliptic modulus of the
/// lattice `Z + tau Z`.
pub fn modular_lambda(tau: &ComplexBall, precision: Precision) -> Result<ComplexBall, ThetaError> {
    modular("lambda", ModularFn::Lambda, tau, precision)
}

/// The modular discriminant `Delta(tau) = eta(tau)^{24}`, the weight-12 cusp
/// form whose `q`-expansion coefficients are Ramanujan's `tau` function.
pub fn modular_discriminant(
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    modular("Delta", ModularFn::Delta, tau, precision)
}

/// The normalised Eisenstein series `E_4, E_6, ..., E_{2*count + 2}`.
///
/// `count` is the number of series wanted, so `count = 2` returns
/// `[E_4, E_6]`. `count = 0` returns an empty vector rather than refusing,
/// because an empty request has an unambiguous empty answer.
pub fn eisenstein_series(
    tau: &ComplexBall,
    count: usize,
    precision: Precision,
) -> Result<Vec<ComplexBall>, ThetaError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    require_upper_half_plane("Eisenstein series", tau)?;
    evaluate_at("Eisenstein series", precision, |bits| {
        backend::eisenstein(tau, count, bits)
    })
}

/// The four classical Jacobi theta functions `[theta_1, theta_2, theta_3,
/// theta_4]` at `(z, tau)`, in FLINT's normalisation: `q = exp(pi i tau)`,
/// `w = exp(pi i z)`, so
/// `theta_3(z, tau) = 1 + 2 sum_{n>=1} q^{n^2} cos(2 n pi z)`.
///
/// In particular `theta_3(0, 0)` is not defined (`tau = 0` is not in the upper
/// half-plane and is refused), while `theta_3(0, tau) -> 1` as `Im(tau) -> inf`
/// and `theta_3(0, i) = pi^{1/4} / Gamma(3/4) = 1.08643481...`.
pub fn jacobi_theta(
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<[ComplexBall; 4], ThetaError> {
    require_upper_half_plane("Jacobi theta", tau)?;
    evaluate_at("Jacobi theta", precision, |bits| {
        backend::jacobi_theta(z, tau, bits)
    })
}

/// The Jacobi theta *constants* `[theta_1(0), theta_2(0), theta_3(0),
/// theta_4(0)]`.
///
/// `theta_1(0, tau)` is identically zero, so the first entry of the result is
/// an enclosure of zero and [`Precision::AccurateTo`] can never be satisfied by
/// this function. Use [`Precision::Bits`] and read the radii.
pub fn jacobi_theta_null(
    tau: &ComplexBall,
    precision: Precision,
) -> Result<[ComplexBall; 4], ThetaError> {
    let zero = ComplexBall::exact_i64(0, 0, tau.precision());
    jacobi_theta(&zero, tau, precision)
}

fn elliptic(
    function: &'static str,
    which: EllipticFn,
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    require_upper_half_plane(function, tau)?;
    evaluate_at(function, precision, |bits| {
        backend::elliptic_fn(which, z, tau, bits)
    })
}

/// The Weierstrass elliptic function `℘(z, tau)` for the lattice `Z + tau Z`.
///
/// `℘` has a double pole at every lattice point; a `z` there (or in a ball that
/// straddles one) comes back with an infinite radius, which
/// [`ComplexBall::is_indeterminate`] reports and
/// [`ComplexBall::value_if_accurate`] refuses on.
pub fn weierstrass_p(
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    elliptic("weierstrass_p", EllipticFn::P, z, tau, precision)
}

/// `℘'(z, tau)`, the derivative of [`weierstrass_p`] with respect to `z`.
pub fn weierstrass_p_prime(
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    elliptic("weierstrass_p_prime", EllipticFn::PPrime, z, tau, precision)
}

/// The Weierstrass zeta function `zeta(z, tau)` (quasi-periodic, `zeta' = -℘`).
pub fn weierstrass_zeta(
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    elliptic("weierstrass_zeta", EllipticFn::Zeta, z, tau, precision)
}

/// The Weierstrass sigma function `sigma(z, tau)`.
pub fn weierstrass_sigma(
    z: &ComplexBall,
    tau: &ComplexBall,
    precision: Precision,
) -> Result<ComplexBall, ThetaError> {
    elliptic("weierstrass_sigma", EllipticFn::Sigma, z, tau, precision)
}

/// The lattice invariants `(g_2, g_3)` of `Z + tau Z`, so that
/// `℘'^2 = 4 ℘^3 - g_2 ℘ - g_3`.
pub fn weierstrass_invariants(
    tau: &ComplexBall,
    precision: Precision,
) -> Result<(ComplexBall, ComplexBall), ThetaError> {
    require_upper_half_plane("weierstrass_invariants", tau)?;
    evaluate_at("weierstrass_invariants", precision, |bits| {
        backend::elliptic_invariants(tau, bits)
    })
}

/// The three roots `(e_1, e_2, e_3)` of `4 x^3 - g_2 x - g_3`, i.e. the values
/// of `℘` at the three half-periods.
pub fn weierstrass_roots(
    tau: &ComplexBall,
    precision: Precision,
) -> Result<[ComplexBall; 3], ThetaError> {
    require_upper_half_plane("weierstrass_roots", tau)?;
    evaluate_at("weierstrass_roots", precision, |bits| {
        backend::elliptic_roots(tau, bits)
    })
}
