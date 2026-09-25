//! The no-FLINT-Arb implementation of [`crate::theta`]'s backend.
//!
//! Selected when `build.rs`'s `flint_arb` symbol probe comes back negative —
//! that is, when `libflint` is a FLINT 2.x that never shipped Arb. The manylinux
//! wheel job still falls back to building FLINT 2.9.0 when `flint-devel` is not
//! installable, so this is a live configuration, not a theoretical one.
//!
//! Every function here refuses with `E-THETA-001`. That is deliberately not the
//! same thing as omitting the module: the Rust API and the Python surface exist
//! identically on every build, so `import alkahest` works, `dir()` agrees, and
//! a caller finds out *why* rather than finding a missing attribute.

use super::ball::{ComplexBall, RealBall};
use super::error::ThetaError;
use super::{CBinop, CUnop, EllipticFn, ModularFn, RBinop, RUnop};

fn unavailable<T>() -> Result<T, ThetaError> {
    Err(ThetaError::BackendUnavailable { capability: "arb" })
}

pub(crate) fn ensure_available() -> Result<(), ThetaError> {
    unavailable()
}

pub(crate) fn ensure_theta_available() -> Result<(), ThetaError> {
    unavailable()
}

pub(crate) const fn riemann_theta_supported() -> bool {
    false
}

pub(crate) fn cb_binop(
    _op: CBinop,
    _a: &ComplexBall,
    _b: &ComplexBall,
    _prec: u32,
) -> Result<ComplexBall, ThetaError> {
    unavailable()
}

pub(crate) fn cb_unop(_op: CUnop, _a: &ComplexBall, _prec: u32) -> Result<ComplexBall, ThetaError> {
    unavailable()
}

pub(crate) fn cb_abs(_a: &ComplexBall, _prec: u32) -> Result<RealBall, ThetaError> {
    unavailable()
}

pub(crate) fn rb_binop(
    _op: RBinop,
    _a: &RealBall,
    _b: &RealBall,
    _prec: u32,
) -> Result<RealBall, ThetaError> {
    unavailable()
}

pub(crate) fn rb_unop(_op: RUnop, _a: &RealBall, _prec: u32) -> Result<RealBall, ThetaError> {
    unavailable()
}

pub(crate) fn real_pi(_prec: u32) -> Result<RealBall, ThetaError> {
    unavailable()
}

pub(crate) fn imag_is_certainly_positive(_tau: &ComplexBall) -> Result<bool, ThetaError> {
    unavailable()
}

pub(crate) fn modular_scalar(
    _which: ModularFn,
    _tau: &ComplexBall,
    _prec: u32,
) -> Result<ComplexBall, ThetaError> {
    unavailable()
}

pub(crate) fn jacobi_theta(
    _z: &ComplexBall,
    _tau: &ComplexBall,
    _prec: u32,
) -> Result<[ComplexBall; 4], ThetaError> {
    unavailable()
}

pub(crate) fn eisenstein(
    _tau: &ComplexBall,
    _len: usize,
    _prec: u32,
) -> Result<Vec<ComplexBall>, ThetaError> {
    unavailable()
}

pub(crate) fn elliptic_fn(
    _which: EllipticFn,
    _z: &ComplexBall,
    _tau: &ComplexBall,
    _prec: u32,
) -> Result<ComplexBall, ThetaError> {
    unavailable()
}

pub(crate) fn elliptic_invariants(
    _tau: &ComplexBall,
    _prec: u32,
) -> Result<(ComplexBall, ComplexBall), ThetaError> {
    unavailable()
}

pub(crate) fn elliptic_roots(
    _tau: &ComplexBall,
    _prec: u32,
) -> Result<[ComplexBall; 3], ThetaError> {
    unavailable()
}

pub(crate) fn siegel_positive_definite(
    _entries: &[ComplexBall],
    _g: usize,
    _prec: u32,
) -> Result<bool, ThetaError> {
    unavailable()
}

pub(crate) fn theta_all(
    _z: &[ComplexBall],
    _tau: &[ComplexBall],
    _g: usize,
    _sqr: bool,
    _prec: u32,
) -> Result<Vec<ComplexBall>, ThetaError> {
    ensure_theta_available()?;
    unavailable()
}

pub(crate) fn theta_one(
    _z: &[ComplexBall],
    _tau: &[ComplexBall],
    _g: usize,
    _ab: u64,
    _prec: u32,
) -> Result<ComplexBall, ThetaError> {
    ensure_theta_available()?;
    unavailable()
}

pub(crate) fn siegel_is_reduced(
    _tau: &[ComplexBall],
    _g: usize,
    _tol_exp: i64,
    _prec: u32,
) -> Result<bool, ThetaError> {
    ensure_theta_available()?;
    unavailable()
}

pub(crate) fn siegel_reduce(
    _tau: &[ComplexBall],
    _g: usize,
    _prec: u32,
) -> Result<(Vec<rug::Integer>, Vec<ComplexBall>), ThetaError> {
    ensure_theta_available()?;
    unavailable()
}

#[allow(dead_code)] // the oracle for `tests::char_parity_matches_flint`
pub(crate) fn char_dot(_a: u64, _b: u64, _g: usize) -> Result<i64, ThetaError> {
    ensure_theta_available()?;
    unavailable()
}
