//! Hand-written FFI to FLINT 3's **Arb** layer — real (`arb_t`) and complex
//! (`acb_t`) balls, their matrices, and the `acb_modular` / `acb_elliptic` /
//! `acb_theta` function libraries built on top of them.
//!
//! FLINT 3.0 absorbed the standalone Arb library, so on a FLINT ≥ 3 install
//! every symbol declared here already lives inside `libflint.so`; no extra
//! `-larb` is needed and none is emitted. FLINT 2.x shipped none of them, which
//! is why this module — and [`crate::flint::acb`] and [`crate::theta`]'s
//! backend — are gated on the `flint_arb` cfg that `build.rs` sets by
//! *probing the symbol table*, not by parsing a version number.
//!
//! # This is not the `ball` module
//!
//! [`crate::ball::ArbBall`] is a **separate, MPFR-backed** implementation of
//! the same mathematical contract, and it is not affected by anything here.
//! This module is a genuine `arb_t` binding; the two coexist deliberately and
//! migrating `ball` onto this one is a separate piece of work.
//!
//! # Struct layout: what is declared and why
//!
//! There are no Arb headers on some of the machines this builds on, so every
//! declaration below is hand-written. Two different risks had to be handled
//! differently:
//!
//! * **`arb_struct` / `acb_struct` / `arf_struct` / `mag_struct` must be
//!   byte-exact.** They are passed *by value inside arrays* — `_acb_vec_init(n)`
//!   returns `n` contiguous `acb_struct`s and this module indexes them with
//!   `ptr.add(i)`. A wrong size is silent memory corruption. They are declared
//!   with their real field structure (mirroring `arf_types.h` / `arb_types.h` /
//!   `acb_types.h`), `const` assertions pin the sizes at compile time, and
//!   [`abi_self_check`] re-derives `size_of::<AcbStruct>()` **at run time** from
//!   FLINT itself by differencing two `acb_mat_entry_ptr` results. Only the
//!   `mid`/`rad` and `real`/`imag` *field offsets* are relied on, and both are
//!   offset 0 or a fixed offset in a struct whose layout has never changed.
//!
//! * **`arb_mat_struct` / `acb_mat_struct` / `fmpz_mat_struct` must not be
//!   declared at all.** These are the structs FLINT 3.1 changed from a row
//!   pointer array to a `stride` — a change with *no size difference*, so
//!   getting it wrong is an integer dereferenced as a pointer. Following the
//!   precedent set by `nmod_mat`/`fq_nmod_mat` in [`crate::flint::ffi`], this
//!   module declares them as **opaque over-sized byte buffers** and reaches
//!   every entry through FLINT's own `acb_mat_entry_ptr` / `arb_mat_entry_ptr`
//!   / `fmpz_mat_entry` accessor functions, which compute the offset inside
//!   the installed library. Row and column counts are tracked on the Rust side
//!   rather than read back out of the struct, so **no field of these three
//!   structs is ever read or written by this crate.**
//!
//! [`abi_self_check`] runs once per process, is called by every public entry
//! point in [`crate::theta`], and turns a layout mismatch into a typed refusal
//! instead of a corrupted heap.

#![allow(dead_code)]

use std::ffi::{c_char, c_int};

use super::ffi::{fmpz, slong};

// ---------------------------------------------------------------------------
// Value types — byte-exact, mirroring arf_types.h / arb_types.h / acb_types.h
// ---------------------------------------------------------------------------

/// `mag_struct` — `{ fmpz exp; ulong man; }`, an unsigned 30-bit-mantissa
/// magnitude used as a ball radius. 16 bytes; opaque payload.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MagStruct {
    _payload: [u64; 2],
}

/// `arf_struct` — `{ fmpz exp; slong size; mantissa_struct d; }` where the
/// mantissa union is two limbs wide (`ARF_NOPTR_LIMBS == 2`). 32 bytes.
///
/// The payload is deliberately opaque: the `size` field encodes whether the
/// mantissa is stored inline or out of line, and nothing in this crate is
/// entitled to that distinction — every read goes through `arf_get_fmpz_2exp`
/// or `arf_get_d`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArfStruct {
    _payload: [u64; 4],
}

/// `arb_struct` — `{ arf_struct mid; mag_struct rad; }`. 48 bytes.
///
/// The two fields are named because this module reads them: `mid` is at offset
/// 0 and `rad` at offset 32, and that has been the layout since Arb 2.x. The
/// *contents* of each remain opaque.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ArbStruct {
    pub mid: ArfStruct,
    pub rad: MagStruct,
}

/// `acb_struct` — `{ arb_struct real; arb_struct imag; }`. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AcbStruct {
    pub real: ArbStruct,
    pub imag: ArbStruct,
}

// Compile-time pins. If a future FLINT widens `ARF_NOPTR_LIMBS` these fail
// here rather than in the allocator.
const _: () = assert!(std::mem::size_of::<MagStruct>() == 16);
const _: () = assert!(std::mem::size_of::<ArfStruct>() == 32);
const _: () = assert!(std::mem::size_of::<ArbStruct>() == 48);
const _: () = assert!(std::mem::size_of::<AcbStruct>() == 96);
const _: () = assert!(std::mem::align_of::<AcbStruct>() == 8);

impl ArbStruct {
    /// An all-zero `arb_struct` is a valid *uninitialised* `arb_t` in FLINT's
    /// model (`arf` zero, `mag` zero), which is exactly what `arb_init` writes.
    pub const fn zeroed() -> Self {
        ArbStruct {
            mid: ArfStruct { _payload: [0; 4] },
            rad: MagStruct { _payload: [0; 2] },
        }
    }
}

impl AcbStruct {
    pub const fn zeroed() -> Self {
        AcbStruct {
            real: ArbStruct::zeroed(),
            imag: ArbStruct::zeroed(),
        }
    }
}

impl ArfStruct {
    pub const fn zeroed() -> Self {
        ArfStruct { _payload: [0; 4] }
    }
}

impl MagStruct {
    pub const fn zeroed() -> Self {
        MagStruct { _payload: [0; 2] }
    }
}

// ---------------------------------------------------------------------------
// Matrix handles — opaque, over-sized, never field-accessed
// ---------------------------------------------------------------------------
//
// READ BEFORE TOUCHING THESE.
//
// `acb_mat_struct`, `arb_mat_struct` and `fmpz_mat_struct` are four
// pointer-sized words on every FLINT this crate supports, but *which* four
// changed in FLINT 3.1: `{entries, r, c, rows: **T}` became
// `{entries, r, c, stride: slong}`. Both are 32 bytes, so a misdeclaration is
// not a size error and not a compile error — it is an integer dereferenced as
// a pointer.
//
// The defence here is stronger than the `flint3_stride` cfg used elsewhere in
// this crate: these buffers are **opaque**. Nothing in `alkahest` reads or
// writes a field of them. Dimensions are carried on the Rust side, and every
// entry address comes from `acb_mat_entry_ptr` / `arb_mat_entry_ptr` /
// `fmpz_mat_entry`, which are real exported functions (not macros) in FLINT
// 3.1+ and compute the offset with whichever field the installed library has.
// `build.rs` refuses to set `flint_arb` unless those accessors are exported.
//
// The buffers are 64 bytes — double the 32 FLINT 3.5.0 writes — and
// `crate::theta::tests` fills the tail with a sentinel and asserts it survives
// `*_init`, so a future FLINT that grows the struct fails a test instead of the
// stack frame.

/// Opaque storage for one `acb_mat_t`. See the module note above.
#[repr(C, align(8))]
pub struct AcbMatBuf(pub [u8; 64]);

/// Opaque storage for one `arb_mat_t`. See the module note above.
#[repr(C, align(8))]
pub struct ArbMatBuf(pub [u8; 64]);

/// Opaque storage for one `fmpz_mat_t`, used only to receive the symplectic
/// matrix `acb_siegel_reduce` produces. See the module note above.
#[repr(C, align(8))]
pub struct FmpzMatBuf(pub [u8; 64]);

impl AcbMatBuf {
    pub const fn sentinel() -> Self {
        AcbMatBuf([0xAA; 64])
    }
}
impl ArbMatBuf {
    pub const fn sentinel() -> Self {
        ArbMatBuf([0xAA; 64])
    }
}
impl FmpzMatBuf {
    pub const fn sentinel() -> Self {
        FmpzMatBuf([0xAA; 64])
    }
}

// ---------------------------------------------------------------------------
// Rounding modes (`arf_rnd_t`)
// ---------------------------------------------------------------------------

/// Toward zero.
pub const ARF_RND_DOWN: c_int = 0;
/// Away from zero.
pub const ARF_RND_UP: c_int = 1;
/// Toward `-inf`.
pub const ARF_RND_FLOOR: c_int = 2;
/// Toward `+inf`.
pub const ARF_RND_CEIL: c_int = 3;
/// To nearest.
pub const ARF_RND_NEAR: c_int = 4;

/// `arb_get_str` flag: print more digits than are justified by the radius.
pub const ARB_STR_MORE: u64 = 1;
/// `arb_get_str` flag: omit the `+/- rad` part.
pub const ARB_STR_NO_RADIUS: u64 = 2;

// ---------------------------------------------------------------------------
// extern "C"
// ---------------------------------------------------------------------------

#[link(name = "flint")]
extern "C" {
    // -- arf ---------------------------------------------------------------
    pub fn arf_init(x: *mut ArfStruct);
    pub fn arf_clear(x: *mut ArfStruct);
    pub fn arf_set_d(x: *mut ArfStruct, v: f64);
    pub fn arf_get_d(x: *const ArfStruct, rnd: c_int) -> f64;
    pub fn arf_set_mag(y: *mut ArfStruct, x: *const MagStruct);
    /// Exact: writes `man` and `exp` with `x == man * 2^exp`. Undefined if `x`
    /// is not finite, so callers must gate on `arb_is_finite` first.
    pub fn arf_get_fmpz_2exp(man: *mut fmpz, exp: *mut fmpz, x: *const ArfStruct);
    pub fn arf_set_fmpz_2exp(x: *mut ArfStruct, man: *const fmpz, exp: *const fmpz);

    // -- mag ---------------------------------------------------------------
    pub fn mag_init(x: *mut MagStruct);
    pub fn mag_clear(x: *mut MagStruct);
    pub fn mag_set_d(z: *mut MagStruct, x: f64);
    pub fn mag_get_d(z: *const MagStruct) -> f64;
    pub fn mag_zero(z: *mut MagStruct);
    pub fn mag_is_inf(x: *const MagStruct) -> c_int;

    // -- arb ---------------------------------------------------------------
    pub fn arb_init(x: *mut ArbStruct);
    pub fn arb_clear(x: *mut ArbStruct);
    pub fn arb_zero(x: *mut ArbStruct);
    pub fn arb_one(x: *mut ArbStruct);
    pub fn arb_indeterminate(x: *mut ArbStruct);
    pub fn arb_set(y: *mut ArbStruct, x: *const ArbStruct);
    pub fn arb_set_d(y: *mut ArbStruct, x: f64);
    pub fn arb_set_si(y: *mut ArbStruct, x: slong);
    pub fn arb_set_ui(y: *mut ArbStruct, x: u64);
    pub fn arb_set_fmpz_2exp(x: *mut ArbStruct, y: *const fmpz, exp: *const fmpz);
    pub fn arb_set_round(z: *mut ArbStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_neg(y: *mut ArbStruct, x: *const ArbStruct);
    pub fn arb_abs(y: *mut ArbStruct, x: *const ArbStruct);
    pub fn arb_add(z: *mut ArbStruct, x: *const ArbStruct, y: *const ArbStruct, prec: slong);
    pub fn arb_sub(z: *mut ArbStruct, x: *const ArbStruct, y: *const ArbStruct, prec: slong);
    pub fn arb_mul(z: *mut ArbStruct, x: *const ArbStruct, y: *const ArbStruct, prec: slong);
    pub fn arb_div(z: *mut ArbStruct, x: *const ArbStruct, y: *const ArbStruct, prec: slong);
    pub fn arb_sqrt(z: *mut ArbStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_pow(z: *mut ArbStruct, x: *const ArbStruct, y: *const ArbStruct, prec: slong);
    pub fn arb_exp(z: *mut ArbStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_log(z: *mut ArbStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_gamma(z: *mut ArbStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_const_pi(z: *mut ArbStruct, prec: slong);
    pub fn arb_mul_2exp_si(y: *mut ArbStruct, x: *const ArbStruct, e: slong);
    pub fn arb_add_error_arf(x: *mut ArbStruct, err: *const ArfStruct);
    pub fn arb_get_mid_arb(z: *mut ArbStruct, x: *const ArbStruct);
    pub fn arb_get_rad_arb(z: *mut ArbStruct, x: *const ArbStruct);
    pub fn arb_get_mag(z: *mut MagStruct, x: *const ArbStruct);
    pub fn arb_get_lbound_arf(u: *mut ArfStruct, x: *const ArbStruct, prec: slong);
    pub fn arb_get_ubound_arf(u: *mut ArfStruct, x: *const ArbStruct, prec: slong);
    /// `floor(log2(|mid| / rad))`, saturating at `ARF_PREC_EXACT` for an exact
    /// ball and at `-ARF_PREC_EXACT` for one whose radius swamps its midpoint.
    pub fn arb_rel_accuracy_bits(x: *const ArbStruct) -> slong;
    pub fn arb_is_finite(x: *const ArbStruct) -> c_int;
    pub fn arb_is_exact(x: *const ArbStruct) -> c_int;
    pub fn arb_is_positive(x: *const ArbStruct) -> c_int;
    pub fn arb_contains_zero(x: *const ArbStruct) -> c_int;
    pub fn arb_contains(x: *const ArbStruct, y: *const ArbStruct) -> c_int;
    pub fn arb_overlaps(x: *const ArbStruct, y: *const ArbStruct) -> c_int;
    pub fn arb_equal(x: *const ArbStruct, y: *const ArbStruct) -> c_int;
    /// Allocates; free with `flint_free`.
    pub fn arb_get_str(x: *const ArbStruct, n: slong, flags: u64) -> *mut c_char;

    // -- acb ---------------------------------------------------------------
    pub fn acb_init(x: *mut AcbStruct);
    pub fn acb_clear(x: *mut AcbStruct);
    pub fn acb_zero(x: *mut AcbStruct);
    pub fn acb_one(x: *mut AcbStruct);
    pub fn acb_onei(x: *mut AcbStruct);
    pub fn acb_indeterminate(x: *mut AcbStruct);
    pub fn acb_set(z: *mut AcbStruct, x: *const AcbStruct);
    pub fn acb_set_d_d(z: *mut AcbStruct, re: f64, im: f64);
    pub fn acb_set_arb_arb(z: *mut AcbStruct, re: *const ArbStruct, im: *const ArbStruct);
    pub fn acb_set_round(z: *mut AcbStruct, x: *const AcbStruct, prec: slong);
    pub fn acb_get_real(re: *mut ArbStruct, z: *const AcbStruct);
    pub fn acb_get_imag(im: *mut ArbStruct, z: *const AcbStruct);
    pub fn acb_neg(z: *mut AcbStruct, x: *const AcbStruct);
    pub fn acb_conj(z: *mut AcbStruct, x: *const AcbStruct);
    pub fn acb_add(z: *mut AcbStruct, x: *const AcbStruct, y: *const AcbStruct, prec: slong);
    pub fn acb_sub(z: *mut AcbStruct, x: *const AcbStruct, y: *const AcbStruct, prec: slong);
    pub fn acb_mul(z: *mut AcbStruct, x: *const AcbStruct, y: *const AcbStruct, prec: slong);
    pub fn acb_div(z: *mut AcbStruct, x: *const AcbStruct, y: *const AcbStruct, prec: slong);
    pub fn acb_sqrt(z: *mut AcbStruct, x: *const AcbStruct, prec: slong);
    pub fn acb_exp(z: *mut AcbStruct, x: *const AcbStruct, prec: slong);
    pub fn acb_log(z: *mut AcbStruct, x: *const AcbStruct, prec: slong);
    pub fn acb_pow(z: *mut AcbStruct, x: *const AcbStruct, y: *const AcbStruct, prec: slong);
    pub fn acb_abs(u: *mut ArbStruct, z: *const AcbStruct, prec: slong);
    pub fn acb_const_pi(x: *mut AcbStruct, prec: slong);
    pub fn acb_mul_2exp_si(z: *mut AcbStruct, x: *const AcbStruct, e: slong);
    pub fn acb_rel_accuracy_bits(x: *const AcbStruct) -> slong;
    pub fn acb_is_finite(x: *const AcbStruct) -> c_int;
    pub fn acb_contains(x: *const AcbStruct, y: *const AcbStruct) -> c_int;
    pub fn acb_contains_zero(x: *const AcbStruct) -> c_int;
    pub fn acb_overlaps(x: *const AcbStruct, y: *const AcbStruct) -> c_int;
    pub fn acb_equal(x: *const AcbStruct, y: *const AcbStruct) -> c_int;

    // -- contiguous vectors ------------------------------------------------
    pub fn _arb_vec_init(n: slong) -> *mut ArbStruct;
    pub fn _arb_vec_clear(v: *mut ArbStruct, n: slong);
    pub fn _acb_vec_init(n: slong) -> *mut AcbStruct;
    pub fn _acb_vec_clear(v: *mut AcbStruct, n: slong);

    // -- matrices (opaque handles; entries only via the accessors) ---------
    pub fn arb_mat_init(mat: *mut ArbMatBuf, r: slong, c: slong);
    pub fn arb_mat_clear(mat: *mut ArbMatBuf);
    pub fn arb_mat_entry_ptr(mat: *mut ArbMatBuf, i: slong, j: slong) -> *mut ArbStruct;
    /// Cholesky factor of a symmetric positive-definite `A`. Returns non-zero
    /// on success, `0` when positive-definiteness could not be established at
    /// the working precision — which is exactly the test this crate uses for
    /// "is `Im(tau)` certainly in the Siegel upper half-space".
    pub fn arb_mat_cho(l: *mut ArbMatBuf, a: *const ArbMatBuf, prec: slong) -> c_int;

    pub fn acb_mat_init(mat: *mut AcbMatBuf, r: slong, c: slong);
    pub fn acb_mat_clear(mat: *mut AcbMatBuf);
    pub fn acb_mat_entry_ptr(mat: *mut AcbMatBuf, i: slong, j: slong) -> *mut AcbStruct;
    pub fn acb_mat_get_imag(im: *mut ArbMatBuf, mat: *const AcbMatBuf);

    /// `fmpz_mat_init` and `fmpz_mat_clear` are **not** redeclared here: they
    /// already exist in [`crate::flint::ffi`] against `FmpzMatStruct`, and two
    /// `extern` declarations of one symbol with different argument types is a
    /// `clashing_extern_declarations` warning (and, under `-D warnings`, an
    /// error). [`crate::flint::acb::IntMat`] calls those, casting its opaque
    /// buffer's address — which is larger and equally aligned — to
    /// `*mut FmpzMatStruct` at the call.
    ///
    // `fmpz_mat_entry` is deliberately NOT declared here. `super::ffi` already
    // declares it over `FmpzMatStruct`, and two `extern "C"` declarations of one
    // symbol with different parameter types is a `clashing_extern_declarations`
    // error under `-D warnings`. The comment this replaces said "nothing else
    // declares it", which was true on the branch this module was written on and
    // false once it met the lattice work. `acb.rs` casts its opaque buffer to
    // `FmpzMatStruct` at the one call site instead; both are `#[repr(C)]` views
    // of the same `fmpz_mat_t`, and only FLINT ever computes an entry address.

    // -- acb_modular -------------------------------------------------------
    pub fn acb_modular_eta(z: *mut AcbStruct, tau: *const AcbStruct, prec: slong);
    pub fn acb_modular_j(z: *mut AcbStruct, tau: *const AcbStruct, prec: slong);
    pub fn acb_modular_lambda(r: *mut AcbStruct, tau: *const AcbStruct, prec: slong);
    pub fn acb_modular_delta(r: *mut AcbStruct, tau: *const AcbStruct, prec: slong);
    /// `E_4, E_6, ..., E_{2*len+2}` written to `r`.
    pub fn acb_modular_eisenstein(
        r: *mut AcbStruct,
        tau: *const AcbStruct,
        len: slong,
        prec: slong,
    );
    /// Jacobi `theta_1 .. theta_4` with `w = exp(pi i z)`, `q = exp(pi i tau)`.
    /// Moves `tau` to the fundamental domain and reduces `z` first.
    pub fn acb_modular_theta(
        theta1: *mut AcbStruct,
        theta2: *mut AcbStruct,
        theta3: *mut AcbStruct,
        theta4: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );

    // -- acb_elliptic ------------------------------------------------------
    pub fn acb_elliptic_p(
        r: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
    pub fn acb_elliptic_p_prime(
        r: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
    pub fn acb_elliptic_zeta(
        r: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
    pub fn acb_elliptic_sigma(
        r: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
    pub fn acb_elliptic_invariants(
        g2: *mut AcbStruct,
        g3: *mut AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
    pub fn acb_elliptic_roots(
        e1: *mut AcbStruct,
        e2: *mut AcbStruct,
        e3: *mut AcbStruct,
        tau: *const AcbStruct,
        prec: slong,
    );
}

// ---------------------------------------------------------------------------
// acb_theta — only declared when the FLINT 3.2+ interface is present
// ---------------------------------------------------------------------------

// Genus-`g` Riemann theta functions and Siegel reduction.
//
// Gated on `flint_acb_theta` because FLINT 3.2 rewrote this interface: before
// it, `acb_theta_all` existed with a different argument list and
// `acb_theta_jet` did not exist at all. `build.rs` only sets the cfg when the
// installed library exports the whole post-rewrite set.
#[cfg(flint_acb_theta)]
#[link(name = "flint")]
extern "C" {
    /// All `4^g` values `theta[a;b](z, tau)` (or their squares when `sqr != 0`),
    /// indexed by the `2g`-bit integer `(a << g) | b`, `a` most significant.
    pub fn acb_theta_all(
        th: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbMatBuf,
        sqr: c_int,
        prec: slong,
    );
    /// The single value `theta[a;b](z, tau)` for the characteristic `ab`.
    pub fn acb_theta_one(
        th: *mut AcbStruct,
        z: *const AcbStruct,
        tau: *const AcbMatBuf,
        ab: u64,
        prec: slong,
    );
    /// `sum_j a_j b_j mod 4` in FLINT's characteristic encoding; its parity is
    /// what decides whether `(a, b)` is an even characteristic.
    pub fn acb_theta_char_dot(a: u64, b: u64, g: slong) -> slong;
    /// A symplectic `mat` with `mat . tau` as reduced as FLINT can make it.
    /// Falls back to the identity when `tau` is unreasonable.
    pub fn acb_siegel_reduce(mat: *mut FmpzMatBuf, tau: *const AcbMatBuf, prec: slong);
    /// Certainly in the reduced domain with tolerance `2^tol_exp`?
    pub fn acb_siegel_is_reduced(tau: *const AcbMatBuf, tol_exp: slong, prec: slong) -> c_int;
    /// `w = (alpha tau + beta)(gamma tau + delta)^-1`.
    pub fn acb_siegel_transform(
        w: *mut AcbMatBuf,
        mat: *const FmpzMatBuf,
        tau: *const AcbMatBuf,
        prec: slong,
    );
    /// Is `mat` in `Sp_{2g}(Z)`?
    pub fn sp2gz_is_correct(mat: *const FmpzMatBuf) -> c_int;
}

// ---------------------------------------------------------------------------
// Runtime ABI self-check
// ---------------------------------------------------------------------------

/// Why the run-time ABI probe rejected the installed FLINT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbiMismatch {
    /// What was measured, in a form that names the offending number.
    pub detail: String,
}

static ABI_OK: std::sync::OnceLock<Result<(), AbiMismatch>> = std::sync::OnceLock::new();

/// Verify, **against the installed FLINT rather than against a header**, that
/// the `arb_struct` / `acb_struct` sizes this module compiled with are the ones
/// the library is using, and that the opaque matrix buffers are large enough.
///
/// The check differences two `acb_mat_entry_ptr` results inside one row: the
/// gap between entry `(0, 0)` and entry `(0, 1)` *is* `sizeof(acb_struct)` as
/// the library sees it, in both the row-pointer and the stride layout. The same
/// is done for `arb_mat`. A sentinel byte pattern past the 32 bytes FLINT
/// 3.5.0 writes is checked to still be intact, so a future FLINT that grows
/// `acb_mat_struct` is caught here instead of scribbling on the stack.
///
/// Runs once per process; every later call reads the memoised verdict.
pub fn abi_self_check() -> Result<(), AbiMismatch> {
    ABI_OK.get_or_init(run_abi_self_check).clone()
}

fn run_abi_self_check() -> Result<(), AbiMismatch> {
    let acb_size = std::mem::size_of::<AcbStruct>();
    let arb_size = std::mem::size_of::<ArbStruct>();

    // SAFETY: both buffers are 64 bytes, twice what any known FLINT writes
    // into an `acb_mat_struct` / `arb_mat_struct`; `*_init` is paired with
    // `*_clear` on every path below, and no field is read directly.
    unsafe {
        let mut m = AcbMatBuf::sentinel();
        acb_mat_init(&mut m, 1, 2);
        let e00 = acb_mat_entry_ptr(&mut m, 0, 0) as usize;
        let e01 = acb_mat_entry_ptr(&mut m, 0, 1) as usize;
        let tail_ok = m.0[32..].iter().all(|&b| b == 0xAA);
        acb_mat_clear(&mut m);
        let measured = e01.wrapping_sub(e00);
        if measured != acb_size {
            return Err(AbiMismatch {
                detail: format!(
                    "FLINT's acb_struct is {measured} bytes, this build assumes {acb_size}"
                ),
            });
        }
        if !tail_ok {
            return Err(AbiMismatch {
                detail: "acb_mat_init wrote past the 32 bytes acb_mat_struct is assumed to \
                         occupy; the opaque buffer in flint::arb needs enlarging"
                    .to_string(),
            });
        }

        let mut r = ArbMatBuf::sentinel();
        arb_mat_init(&mut r, 1, 2);
        let f00 = arb_mat_entry_ptr(&mut r, 0, 0) as usize;
        let f01 = arb_mat_entry_ptr(&mut r, 0, 1) as usize;
        let tail_ok = r.0[32..].iter().all(|&b| b == 0xAA);
        arb_mat_clear(&mut r);
        let measured = f01.wrapping_sub(f00);
        if measured != arb_size {
            return Err(AbiMismatch {
                detail: format!(
                    "FLINT's arb_struct is {measured} bytes, this build assumes {arb_size}"
                ),
            });
        }
        if !tail_ok {
            return Err(AbiMismatch {
                detail: "arb_mat_init wrote past the 32 bytes arb_mat_struct is assumed to \
                         occupy; the opaque buffer in flint::arb needs enlarging"
                    .to_string(),
            });
        }

        // The `mid` / `rad` field offsets are the other thing this module
        // relies on. Setting a ball to exactly 1.5 and reading its midpoint
        // back through `&x.mid` proves the offset is right: a wrong offset
        // reads the radius (0) or the neighbouring struct.
        let mut x = ArbStruct::zeroed();
        arb_init(&mut x);
        arb_set_d(&mut x, 1.5);
        let mid = arf_get_d(&x.mid, ARF_RND_NEAR);
        let rad = mag_get_d(&x.rad);
        arb_clear(&mut x);
        if mid != 1.5 || rad != 0.0 {
            return Err(AbiMismatch {
                detail: format!(
                    "arb_struct field offsets disagree with FLINT: read mid={mid}, rad={rad} \
                     from a ball set to exactly 1.5"
                ),
            });
        }

        // And the acb real/imag offsets.
        let mut z = AcbStruct::zeroed();
        acb_init(&mut z);
        acb_set_d_d(&mut z, 2.25, -3.5);
        let re = arf_get_d(&z.real.mid, ARF_RND_NEAR);
        let im = arf_get_d(&z.imag.mid, ARF_RND_NEAR);
        acb_clear(&mut z);
        if re != 2.25 || im != -3.5 {
            return Err(AbiMismatch {
                detail: format!(
                    "acb_struct field offsets disagree with FLINT: read re={re}, im={im} \
                     from 2.25 - 3.5i"
                ),
            });
        }
    }

    Ok(())
}
