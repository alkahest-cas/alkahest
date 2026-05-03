//! Raw unsafe FFI bindings to FLINT 2.8.4.
#![allow(dead_code)]
//!
//! All raw pointers live only inside `mod ffi` — nothing outside this module
//! should touch them directly.
//!
//! Type mapping (64-bit Linux):
//!   slong = mp_limb_signed_t = long = i64
//!   ulong = mp_limb_t        = u64
//!   fmpz  = slong (tagged pointer: low bit set → points to GMP mpz)
//!   fmpz_t = fmpz[1] → decays to *mut fmpz in function signatures

use std::ffi::{c_int, c_void};
use std::os::raw::c_char;

// ---------------------------------------------------------------------------
// fmpz_mpoly — sparse multivariate polynomials over Z
// ---------------------------------------------------------------------------
//
// Struct sizes verified with a C test on 64-bit Linux with FLINT 2.8.4:
//   sizeof(fmpz_mpoly_ctx_t) = 608
//   sizeof(fmpz_mpoly_t)     = 40
//
// We use opaque byte arrays aligned to u64 to hold the C structs on the
// Rust stack without exposing their internal layout.

/// Opaque storage for a single `fmpz_mpoly_ctx_t` (608 bytes on 64-bit).
#[repr(C, align(8))]
pub struct FmpzMPolyCtxBuf(pub [u8; 608]);

/// Opaque storage for a single `fmpz_mpoly_t` (40 bytes on 64-bit).
#[repr(C, align(8))]
pub struct FmpzMPolyBuf(pub [u8; 40]);

unsafe impl Send for FmpzMPolyCtxBuf {}
unsafe impl Sync for FmpzMPolyCtxBuf {}
unsafe impl Send for FmpzMPolyBuf {}
unsafe impl Sync for FmpzMPolyBuf {}

#[allow(non_camel_case_types)]
pub type slong = i64;
#[allow(non_camel_case_types)]
pub type ulong = u64;
/// `fmpz` is either an inline signed integer or a tagged GMP pointer.
/// It must be treated as opaque storage; never inspect the bits directly.
#[allow(non_camel_case_types)]
pub type fmpz = slong;

/// C layout of `fmpz_poly_struct`. Functions take `*mut fmpz_poly_struct`
/// because `fmpz_poly_t = fmpz_poly_struct[1]` decays to a pointer.
#[repr(C)]
pub struct FmpzPolyStruct {
    pub coeffs: *mut fmpz,
    pub alloc: slong,
    pub length: slong,
}

// SAFETY: FLINT integers and polynomials are safe to send across threads
// (they own their memory and have no thread-local state).
unsafe impl Send for FmpzPolyStruct {}
unsafe impl Sync for FmpzPolyStruct {}

#[link(name = "flint")]
extern "C" {
    // -----------------------------------------------------------------------
    // Memory
    // -----------------------------------------------------------------------
    pub fn flint_free(ptr: *mut c_void);

    // -----------------------------------------------------------------------
    // fmpz — arbitrary-precision integers
    // -----------------------------------------------------------------------
    pub fn fmpz_init(f: *mut fmpz);
    pub fn fmpz_clear(f: *mut fmpz);
    pub fn fmpz_set(f: *mut fmpz, g: *const fmpz);
    pub fn fmpz_set_si(f: *mut fmpz, val: slong);
    pub fn fmpz_get_si(f: *const fmpz) -> slong;
    pub fn fmpz_get_str(str_: *mut c_char, b: c_int, f: *const fmpz) -> *mut c_char;
    /// Parse a string into an fmpz. Returns 0 on success, -1 on failure.
    pub fn fmpz_set_str(f: *mut fmpz, str_: *const c_char, b: c_int) -> c_int;
    pub fn fmpz_equal(f: *const fmpz, g: *const fmpz) -> c_int;
    pub fn fmpz_add(f: *mut fmpz, g: *const fmpz, h: *const fmpz);
    pub fn fmpz_sub(f: *mut fmpz, g: *const fmpz, h: *const fmpz);
    pub fn fmpz_mul(f: *mut fmpz, g: *const fmpz, h: *const fmpz);
    /// Truncated (toward-zero) integer division.
    pub fn fmpz_tdiv_q(f: *mut fmpz, g: *const fmpz, h: *const fmpz);
    /// Truncated division: sets q = trunc(g/h) and r = g - q*h simultaneously.
    pub fn fmpz_tdiv_qr(q: *mut fmpz, r: *mut fmpz, g: *const fmpz, h: *const fmpz);
    pub fn fmpz_neg(f: *mut fmpz, g: *const fmpz);
    pub fn fmpz_gcd(f: *mut fmpz, g: *const fmpz, h: *const fmpz);
    pub fn fmpz_pow_ui(f: *mut fmpz, g: *const fmpz, x: ulong);

    // -----------------------------------------------------------------------
    // fmpz_poly — dense univariate polynomials over Z
    // -----------------------------------------------------------------------
    pub fn fmpz_poly_init(poly: *mut FmpzPolyStruct);
    pub fn fmpz_poly_clear(poly: *mut FmpzPolyStruct);
    pub fn fmpz_poly_set(dst: *mut FmpzPolyStruct, src: *const FmpzPolyStruct);
    pub fn fmpz_poly_set_coeff_si(poly: *mut FmpzPolyStruct, n: slong, x: slong);
    pub fn fmpz_poly_get_coeff_si(poly: *const FmpzPolyStruct, n: slong) -> slong;
    pub fn fmpz_poly_set_coeff_fmpz(poly: *mut FmpzPolyStruct, n: slong, x: *const fmpz);
    pub fn fmpz_poly_get_coeff_fmpz(x: *mut fmpz, poly: *const FmpzPolyStruct, n: slong);
    pub fn fmpz_poly_length(poly: *const FmpzPolyStruct) -> slong;
    pub fn fmpz_poly_degree(poly: *const FmpzPolyStruct) -> slong;
    pub fn fmpz_poly_add(
        res: *mut FmpzPolyStruct,
        a: *const FmpzPolyStruct,
        b: *const FmpzPolyStruct,
    );
    pub fn fmpz_poly_sub(
        res: *mut FmpzPolyStruct,
        a: *const FmpzPolyStruct,
        b: *const FmpzPolyStruct,
    );
    pub fn fmpz_poly_mul(
        res: *mut FmpzPolyStruct,
        a: *const FmpzPolyStruct,
        b: *const FmpzPolyStruct,
    );
    pub fn fmpz_poly_pow(res: *mut FmpzPolyStruct, poly: *const FmpzPolyStruct, e: ulong);
    pub fn fmpz_poly_gcd(
        res: *mut FmpzPolyStruct,
        a: *const FmpzPolyStruct,
        b: *const FmpzPolyStruct,
    );
    /// Exact polynomial division: sets Q = A / B, assuming B | A.
    pub fn fmpz_poly_div(
        q: *mut FmpzPolyStruct,
        a: *const FmpzPolyStruct,
        b: *const FmpzPolyStruct,
    );
    pub fn fmpz_poly_equal(a: *const FmpzPolyStruct, b: *const FmpzPolyStruct) -> c_int;
    /// Allocates and returns a human-readable string. Caller must free with `flint_free`.
    pub fn fmpz_poly_get_str_pretty(poly: *const FmpzPolyStruct, x: *const c_char) -> *mut c_char;

    // -----------------------------------------------------------------------
    // fmpz_mpoly — sparse multivariate polynomials over Z
    // -----------------------------------------------------------------------

    /// Initialise a multivariate polynomial context for `nvars` variables with
    /// lexicographic ordering (ord = 0 = ORD_LEX).
    pub fn fmpz_mpoly_ctx_init(ctx: *mut FmpzMPolyCtxBuf, nvars: slong, ord: c_int);
    pub fn fmpz_mpoly_ctx_clear(ctx: *mut FmpzMPolyCtxBuf);

    pub fn fmpz_mpoly_init(A: *mut FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf);
    pub fn fmpz_mpoly_clear(A: *mut FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf);

    pub fn fmpz_mpoly_is_zero(A: *const FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf) -> c_int;
    pub fn fmpz_mpoly_length(A: *const FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf) -> slong;

    /// Push a term `coeff * x^exp[0] * y^exp[1] * ...`.
    /// `exp` must have length equal to `nvars`.
    pub fn fmpz_mpoly_push_term_fmpz_ui(
        A: *mut FmpzMPolyBuf,
        c: *const fmpz,
        exp: *const u64,
        ctx: *const FmpzMPolyCtxBuf,
    );

    /// Sort terms and combine like monomials (call after push_term_* sequence).
    pub fn fmpz_mpoly_sort_terms(A: *mut FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf);
    pub fn fmpz_mpoly_combine_like_terms(A: *mut FmpzMPolyBuf, ctx: *const FmpzMPolyCtxBuf);

    /// Retrieve the coefficient of term `i` as an fmpz (caller must init/clear).
    pub fn fmpz_mpoly_get_term_coeff_fmpz(
        c: *mut fmpz,
        A: *const FmpzMPolyBuf,
        i: slong,
        ctx: *const FmpzMPolyCtxBuf,
    );

    /// Retrieve the exponent vector of term `i` as an array of u64.
    /// `exp` must have length at least `nvars`.
    pub fn fmpz_mpoly_get_term_exp_ui(
        exp: *mut u64,
        A: *const FmpzMPolyBuf,
        i: slong,
        ctx: *const FmpzMPolyCtxBuf,
    );

    /// G = gcd(A, B).  Returns 1 on success, 0 if GCD computation failed.
    pub fn fmpz_mpoly_gcd(
        G: *mut FmpzMPolyBuf,
        A: *const FmpzMPolyBuf,
        B: *const FmpzMPolyBuf,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> c_int;

    /// Q = A / B assuming B | A.  Returns 1 if exact, 0 otherwise.
    pub fn fmpz_mpoly_divides(
        Q: *mut FmpzMPolyBuf,
        A: *const FmpzMPolyBuf,
        B: *const FmpzMPolyBuf,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // fmpz_mpoly — resultant
    // -----------------------------------------------------------------------

    /// Compute the resultant of A and B with respect to variable `var`.
    /// Returns 1 on success, 0 on failure.
    pub fn fmpz_mpoly_resultant(
        R: *mut FmpzMPolyBuf,
        A: *const FmpzMPolyBuf,
        B: *const FmpzMPolyBuf,
        var: slong,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // fmpz_poly — resultant and pseudo-division (for subresultant PRS)
    // -----------------------------------------------------------------------

    /// Compute the resultant of `a` and `b`, stored as an `fmpz`.
    pub fn fmpz_poly_resultant(res: *mut fmpz, a: *const FmpzPolyStruct, b: *const FmpzPolyStruct);

    /// Pseudo-division: sets Q, R, and d such that lc(B)^d * A = Q*B + R.
    pub fn fmpz_poly_pseudo_divrem(
        Q: *mut FmpzPolyStruct,
        R: *mut FmpzPolyStruct,
        d: *mut ulong,
        A: *const FmpzPolyStruct,
        B: *const FmpzPolyStruct,
    );

    /// Negate: res = -poly.
    pub fn fmpz_poly_neg(res: *mut FmpzPolyStruct, poly: *const FmpzPolyStruct);

    /// Scalar multiply: res = poly * x.
    pub fn fmpz_poly_scalar_mul_fmpz(
        res: *mut FmpzPolyStruct,
        poly: *const FmpzPolyStruct,
        x: *const fmpz,
    );

    /// Exact scalar divide: res = poly / x (assumes x divides all coefficients).
    pub fn fmpz_poly_scalar_divexact_fmpz(
        res: *mut FmpzPolyStruct,
        poly: *const FmpzPolyStruct,
        x: *const fmpz,
    );
}
