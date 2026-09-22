//! The FLINT-backed implementation of [`crate::theta`].
//!
//! Compiled only when `build.rs` has proved, by reading `libflint`'s symbol
//! table, that the Arb layer is present (`flint_arb`); the `acb_theta` entry
//! points are further gated on `flint_acb_theta`. `backend_stub.rs` provides
//! the same signatures for a FLINT that has neither, so that
//! [`crate::theta`]'s public API — and therefore the Python surface — exists on
//! every build and refuses with `E-THETA-001` rather than failing to link.
//!
//! Nothing outside this file touches a FLINT pointer.

use crate::flint::acb::{Acb, AcbMat, AcbVec, Arb};
use crate::flint::arb;

use super::ball::{ComplexBall, RealBall};
use super::error::ThetaError;
use super::{CBinop, CUnop, EllipticFn, ModularFn, RBinop, RUnop};

// ---------------------------------------------------------------------------
// Availability
// ---------------------------------------------------------------------------

/// Arb/Acb present, and its struct layout agrees with what this build assumes.
pub(crate) fn ensure_available() -> Result<(), ThetaError> {
    arb::abi_self_check().map_err(|e| ThetaError::AbiMismatch { detail: e.detail })
}

/// As [`ensure_available`], and the FLINT 3.2+ `acb_theta` interface too.
pub(crate) fn ensure_theta_available() -> Result<(), ThetaError> {
    ensure_available()?;
    #[cfg(not(flint_acb_theta))]
    {
        return Err(ThetaError::BackendUnavailable {
            capability: "acb_theta",
        });
    }
    #[cfg(flint_acb_theta)]
    Ok(())
}

/// Is the genus-`g` Riemann theta path compiled in at all?
pub(crate) const fn riemann_theta_supported() -> bool {
    cfg!(flint_acb_theta)
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

fn real_to_arb(b: &RealBall) -> Result<Arb, ThetaError> {
    if b.is_indeterminate() {
        return Ok(Arb::indeterminate());
    }
    Arb::from_mid_rad(b.midpoint(), b.radius()).ok_or(ThetaError::NotRepresentable {
        what: "midpoint or radius",
    })
}

fn arb_to_real(x: &Arb, prec: u32) -> Result<RealBall, ThetaError> {
    if !x.is_finite() {
        return Ok(RealBall::indeterminate(prec));
    }
    let mid = x
        .midpoint_rug()
        .ok_or(ThetaError::NotRepresentable { what: "midpoint" })?;
    let rad = x
        .radius_rug()
        .ok_or(ThetaError::NotRepresentable { what: "radius" })?;
    Ok(RealBall::from_mid_rad(mid, rad, prec))
}

fn to_acb(b: &ComplexBall) -> Result<Acb, ThetaError> {
    let re = real_to_arb(b.real())?;
    let im = real_to_arb(b.imag())?;
    Ok(Acb::from_parts(&re, &im))
}

fn from_acb(z: &Acb, prec: u32) -> Result<ComplexBall, ThetaError> {
    if !z.is_finite() {
        return Ok(ComplexBall::indeterminate(prec));
    }
    Ok(ComplexBall::from_parts(
        arb_to_real(&z.real(), prec)?,
        arb_to_real(&z.imag(), prec)?,
    ))
}

// ---------------------------------------------------------------------------
// Ball arithmetic
// ---------------------------------------------------------------------------

pub(crate) fn cb_binop(
    op: CBinop,
    a: &ComplexBall,
    b: &ComplexBall,
    prec: u32,
) -> Result<ComplexBall, ThetaError> {
    ensure_available()?;
    let x = to_acb(a)?;
    let y = to_acb(b)?;
    let p = i64::from(prec);
    let r = match op {
        CBinop::Add => x.add(&y, p),
        CBinop::Sub => x.sub(&y, p),
        CBinop::Mul => x.mul(&y, p),
        CBinop::Div => x.div(&y, p),
        CBinop::Pow => x.pow(&y, p),
    };
    from_acb(&r, prec)
}

pub(crate) fn cb_unop(op: CUnop, a: &ComplexBall, prec: u32) -> Result<ComplexBall, ThetaError> {
    ensure_available()?;
    let x = to_acb(a)?;
    let p = i64::from(prec);
    let r = match op {
        CUnop::Neg => x.neg(),
        CUnop::Conj => x.conj(),
        CUnop::Sqrt => x.sqrt(p),
        CUnop::Exp => x.exp(p),
        CUnop::Log => x.log(p),
    };
    from_acb(&r, prec)
}

pub(crate) fn cb_abs(a: &ComplexBall, prec: u32) -> Result<RealBall, ThetaError> {
    ensure_available()?;
    let x = to_acb(a)?;
    arb_to_real(&x.abs(i64::from(prec)), prec)
}

pub(crate) fn rb_binop(
    op: RBinop,
    a: &RealBall,
    b: &RealBall,
    prec: u32,
) -> Result<RealBall, ThetaError> {
    ensure_available()?;
    let x = real_to_arb(a)?;
    let y = real_to_arb(b)?;
    let p = i64::from(prec);
    let r = match op {
        RBinop::Add => x.add(&y, p),
        RBinop::Sub => x.sub(&y, p),
        RBinop::Mul => x.mul(&y, p),
        RBinop::Div => x.div(&y, p),
        RBinop::Pow => x.pow(&y, p),
    };
    arb_to_real(&r, prec)
}

pub(crate) fn rb_unop(op: RUnop, a: &RealBall, prec: u32) -> Result<RealBall, ThetaError> {
    ensure_available()?;
    let x = real_to_arb(a)?;
    let p = i64::from(prec);
    let r = match op {
        RUnop::Neg => x.neg(),
        RUnop::Abs => x.abs(),
        RUnop::Sqrt => x.sqrt(p),
        RUnop::Exp => x.exp(p),
        RUnop::Log => x.log(p),
        RUnop::Gamma => x.gamma(p),
    };
    arb_to_real(&r, prec)
}

pub(crate) fn real_pi(prec: u32) -> Result<RealBall, ThetaError> {
    ensure_available()?;
    arb_to_real(&Arb::pi(i64::from(prec)), prec)
}

// ---------------------------------------------------------------------------
// Genus 1: modular and elliptic functions
// ---------------------------------------------------------------------------

/// Is `Im(tau)` certainly `> 0`?
pub(crate) fn imag_is_certainly_positive(tau: &ComplexBall) -> Result<bool, ThetaError> {
    ensure_available()?;
    let z = to_acb(tau)?;
    Ok(z.imag().is_positive())
}

pub(crate) fn modular_scalar(
    which: ModularFn,
    tau: &ComplexBall,
    prec: u32,
) -> Result<ComplexBall, ThetaError> {
    ensure_available()?;
    let t = to_acb(tau)?;
    let mut out = Acb::new();
    let p = i64::from(prec);
    // SAFETY: `out` is an initialised `acb_t`, `t` an initialised `acb_t`, and
    // each of these entry points writes only its first argument.
    unsafe {
        match which {
            ModularFn::Eta => arb::acb_modular_eta(out.as_mut_ptr(), t.as_ptr(), p),
            ModularFn::JInvariant => arb::acb_modular_j(out.as_mut_ptr(), t.as_ptr(), p),
            ModularFn::Lambda => arb::acb_modular_lambda(out.as_mut_ptr(), t.as_ptr(), p),
            ModularFn::Delta => arb::acb_modular_delta(out.as_mut_ptr(), t.as_ptr(), p),
        }
    }
    from_acb(&out, prec)
}

/// `theta_1 .. theta_4` in FLINT's `acb_modular` normalisation:
/// `w = exp(pi i z)`, `q = exp(pi i tau)`.
pub(crate) fn jacobi_theta(
    z: &ComplexBall,
    tau: &ComplexBall,
    prec: u32,
) -> Result<[ComplexBall; 4], ThetaError> {
    ensure_available()?;
    let zz = to_acb(z)?;
    let tt = to_acb(tau)?;
    let mut t1 = Acb::new();
    let mut t2 = Acb::new();
    let mut t3 = Acb::new();
    let mut t4 = Acb::new();
    // SAFETY: four distinct initialised outputs, two initialised inputs; FLINT
    // documents that aliasing between them is not permitted and there is none.
    unsafe {
        arb::acb_modular_theta(
            t1.as_mut_ptr(),
            t2.as_mut_ptr(),
            t3.as_mut_ptr(),
            t4.as_mut_ptr(),
            zz.as_ptr(),
            tt.as_ptr(),
            i64::from(prec),
        );
    }
    Ok([
        from_acb(&t1, prec)?,
        from_acb(&t2, prec)?,
        from_acb(&t3, prec)?,
        from_acb(&t4, prec)?,
    ])
}

/// `E_4, E_6, ..., E_{2*len+2}`.
pub(crate) fn eisenstein(
    tau: &ComplexBall,
    len: usize,
    prec: u32,
) -> Result<Vec<ComplexBall>, ThetaError> {
    ensure_available()?;
    let t = to_acb(tau)?;
    let mut out = AcbVec::new(len);
    // SAFETY: `out` holds exactly `len` initialised `acb_struct`s, which is
    // what the `len` argument promises FLINT.
    unsafe {
        arb::acb_modular_eisenstein(out.as_mut_ptr(), t.as_ptr(), len as i64, i64::from(prec));
    }
    (0..len).map(|i| from_acb(&out.get(i), prec)).collect()
}

pub(crate) fn elliptic_fn(
    which: EllipticFn,
    z: &ComplexBall,
    tau: &ComplexBall,
    prec: u32,
) -> Result<ComplexBall, ThetaError> {
    ensure_available()?;
    let zz = to_acb(z)?;
    let tt = to_acb(tau)?;
    let mut out = Acb::new();
    let p = i64::from(prec);
    // SAFETY: as for `modular_scalar`.
    unsafe {
        match which {
            EllipticFn::P => arb::acb_elliptic_p(out.as_mut_ptr(), zz.as_ptr(), tt.as_ptr(), p),
            EllipticFn::PPrime => {
                arb::acb_elliptic_p_prime(out.as_mut_ptr(), zz.as_ptr(), tt.as_ptr(), p)
            }
            EllipticFn::Zeta => {
                arb::acb_elliptic_zeta(out.as_mut_ptr(), zz.as_ptr(), tt.as_ptr(), p)
            }
            EllipticFn::Sigma => {
                arb::acb_elliptic_sigma(out.as_mut_ptr(), zz.as_ptr(), tt.as_ptr(), p)
            }
        }
    }
    from_acb(&out, prec)
}

pub(crate) fn elliptic_invariants(
    tau: &ComplexBall,
    prec: u32,
) -> Result<(ComplexBall, ComplexBall), ThetaError> {
    ensure_available()?;
    let t = to_acb(tau)?;
    let mut g2 = Acb::new();
    let mut g3 = Acb::new();
    // SAFETY: two distinct initialised outputs, one initialised input.
    unsafe {
        arb::acb_elliptic_invariants(
            g2.as_mut_ptr(),
            g3.as_mut_ptr(),
            t.as_ptr(),
            i64::from(prec),
        );
    }
    Ok((from_acb(&g2, prec)?, from_acb(&g3, prec)?))
}

pub(crate) fn elliptic_roots(tau: &ComplexBall, prec: u32) -> Result<[ComplexBall; 3], ThetaError> {
    ensure_available()?;
    let t = to_acb(tau)?;
    let mut e1 = Acb::new();
    let mut e2 = Acb::new();
    let mut e3 = Acb::new();
    // SAFETY: three distinct initialised outputs, one initialised input.
    unsafe {
        arb::acb_elliptic_roots(
            e1.as_mut_ptr(),
            e2.as_mut_ptr(),
            e3.as_mut_ptr(),
            t.as_ptr(),
            i64::from(prec),
        );
    }
    Ok([
        from_acb(&e1, prec)?,
        from_acb(&e2, prec)?,
        from_acb(&e3, prec)?,
    ])
}

// ---------------------------------------------------------------------------
// Genus g: Riemann theta
// ---------------------------------------------------------------------------

/// Build an `acb_mat_t` from a row-major slice of `g*g` balls.
fn to_acb_mat(entries: &[ComplexBall], g: usize) -> Result<AcbMat, ThetaError> {
    let mut m = AcbMat::new(g, g);
    for i in 0..g {
        for j in 0..g {
            let v = to_acb(&entries[i * g + j])?;
            m.set_entry(i, j, &v);
        }
    }
    Ok(m)
}

fn to_acb_vec(entries: &[ComplexBall]) -> Result<AcbVec, ThetaError> {
    let mut v = AcbVec::new(entries.len());
    for (i, e) in entries.iter().enumerate() {
        v.set(i, &to_acb(e)?);
    }
    Ok(v)
}

/// Is `Im(tau)` certainly positive definite — i.e. is `tau` certainly in `H_g`?
pub(crate) fn siegel_positive_definite(
    entries: &[ComplexBall],
    g: usize,
    prec: u32,
) -> Result<bool, ThetaError> {
    ensure_available()?;
    let m = to_acb_mat(entries, g)?;
    let im = m.imag();
    Ok(im.is_certainly_positive_definite(i64::from(prec)))
}

#[cfg(flint_acb_theta)]
pub(crate) fn theta_all(
    z: &[ComplexBall],
    tau: &[ComplexBall],
    g: usize,
    sqr: bool,
    prec: u32,
) -> Result<Vec<ComplexBall>, ThetaError> {
    ensure_theta_available()?;
    let n = 1usize << (2 * g);
    let zv = to_acb_vec(z)?;
    let mut tm = to_acb_mat(tau, g)?;
    let mut out = AcbVec::new(n);
    // SAFETY: `out` holds `4^g` initialised slots, which is the length
    // `acb_theta_all` writes for genus `g`; `zv` holds `g`, which is the length
    // it reads. `tm` is a `g x g` `acb_mat_t`.
    unsafe {
        arb::acb_theta_all(
            out.as_mut_ptr(),
            zv.as_ptr(),
            tm.as_mut_ptr(),
            i32::from(sqr),
            i64::from(prec),
        );
    }
    (0..n).map(|i| from_acb(&out.get(i), prec)).collect()
}

#[cfg(not(flint_acb_theta))]
pub(crate) fn theta_all(
    _z: &[ComplexBall],
    _tau: &[ComplexBall],
    _g: usize,
    _sqr: bool,
    _prec: u32,
) -> Result<Vec<ComplexBall>, ThetaError> {
    Err(ThetaError::BackendUnavailable {
        capability: "acb_theta",
    })
}

#[cfg(flint_acb_theta)]
pub(crate) fn theta_one(
    z: &[ComplexBall],
    tau: &[ComplexBall],
    g: usize,
    ab: u64,
    prec: u32,
) -> Result<ComplexBall, ThetaError> {
    ensure_theta_available()?;
    let zv = to_acb_vec(z)?;
    let mut tm = to_acb_mat(tau, g)?;
    // `acb_theta_one` takes an `acb_ptr`, not an `acb_t`: a one-element vector
    // is the right shape and keeps the call honest about that.
    let mut out = AcbVec::new(1);
    // SAFETY: as for `theta_all`, with a single output slot.
    unsafe {
        arb::acb_theta_one(
            out.as_mut_ptr(),
            zv.as_ptr(),
            tm.as_mut_ptr(),
            ab,
            i64::from(prec),
        );
    }
    from_acb(&out.get(0), prec)
}

#[cfg(not(flint_acb_theta))]
pub(crate) fn theta_one(
    _z: &[ComplexBall],
    _tau: &[ComplexBall],
    _g: usize,
    _ab: u64,
    _prec: u32,
) -> Result<ComplexBall, ThetaError> {
    Err(ThetaError::BackendUnavailable {
        capability: "acb_theta",
    })
}

#[cfg(flint_acb_theta)]
pub(crate) fn siegel_is_reduced(
    tau: &[ComplexBall],
    g: usize,
    tol_exp: i64,
    prec: u32,
) -> Result<bool, ThetaError> {
    ensure_theta_available()?;
    let mut tm = to_acb_mat(tau, g)?;
    // SAFETY: `tm` is an initialised `g x g` `acb_mat_t`; the call only reads it.
    Ok(unsafe { arb::acb_siegel_is_reduced(tm.as_mut_ptr(), tol_exp, i64::from(prec)) != 0 })
}

#[cfg(not(flint_acb_theta))]
pub(crate) fn siegel_is_reduced(
    _tau: &[ComplexBall],
    _g: usize,
    _tol_exp: i64,
    _prec: u32,
) -> Result<bool, ThetaError> {
    Err(ThetaError::BackendUnavailable {
        capability: "acb_theta",
    })
}

/// `(symplectic matrix as 2g x 2g integers, reduced period matrix)`.
#[cfg(flint_acb_theta)]
pub(crate) fn siegel_reduce(
    tau: &[ComplexBall],
    g: usize,
    prec: u32,
) -> Result<(Vec<rug::Integer>, Vec<ComplexBall>), ThetaError> {
    use crate::flint::acb::IntMat;
    ensure_theta_available()?;
    let mut tm = to_acb_mat(tau, g)?;
    let mut mat = IntMat::new(2 * g, 2 * g);
    let mut w = AcbMat::new(g, g);
    // SAFETY: `mat` is a `2g x 2g` `fmpz_mat_t`, which is the shape
    // `acb_siegel_reduce` writes for genus `g`; `w` is `g x g`, the shape
    // `acb_siegel_transform` writes.
    unsafe {
        arb::acb_siegel_reduce(mat.as_mut_ptr(), tm.as_mut_ptr(), i64::from(prec));
        arb::acb_siegel_transform(
            w.as_mut_ptr(),
            mat.as_ptr(),
            tm.as_mut_ptr(),
            i64::from(prec),
        );
    }
    let mut ints = Vec::with_capacity(4 * g * g);
    for i in 0..2 * g {
        for j in 0..2 * g {
            ints.push(mat.get_entry(i, j));
        }
    }
    let mut reduced = Vec::with_capacity(g * g);
    for i in 0..g {
        for j in 0..g {
            reduced.push(from_acb(&w.get_entry(i, j), prec)?);
        }
    }
    Ok((ints, reduced))
}

#[cfg(not(flint_acb_theta))]
pub(crate) fn siegel_reduce(
    _tau: &[ComplexBall],
    _g: usize,
    _prec: u32,
) -> Result<(Vec<rug::Integer>, Vec<ComplexBall>), ThetaError> {
    Err(ThetaError::BackendUnavailable {
        capability: "acb_theta",
    })
}

/// FLINT's `acb_theta_char_dot`: `sum_j a_j b_j` over the `g` low bits of each.
///
/// Nothing in the public API calls this: characteristic parity is computed in
/// pure Rust so that it works on a build with no `acb_theta`. It is kept as the
/// **oracle** that `tests::char_parity_matches_flint` checks that pure-Rust
/// version against, for every characteristic up to genus 3.
#[allow(dead_code)]
#[cfg(flint_acb_theta)]
pub(crate) fn char_dot(a: u64, b: u64, g: usize) -> Result<i64, ThetaError> {
    ensure_theta_available()?;
    // SAFETY: a pure integer function with no pointer arguments.
    Ok(unsafe { arb::acb_theta_char_dot(a, b, g as i64) })
}

#[cfg(not(flint_acb_theta))]
#[allow(dead_code)]
pub(crate) fn char_dot(_a: u64, _b: u64, _g: usize) -> Result<i64, ThetaError> {
    Err(ThetaError::BackendUnavailable {
        capability: "acb_theta",
    })
}
