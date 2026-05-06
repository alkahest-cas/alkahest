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

/// `fmpz_poly_factor_struct` — FLINT factorization container (`fmpz_poly_factor_t`).
#[repr(C)]
pub struct FmpzPolyFactorStruct {
    pub c: fmpz,
    pub p: *mut FmpzPolyStruct,
    pub exp: *mut slong,
    pub num: slong,
    pub alloc: slong,
}

/// FLINT `nmod_t` — modulus data for `nmod_poly`.
#[repr(C)]
pub struct NmodStruct {
    pub n: ulong,
    pub ninv: ulong,
    pub norm: ulong,
}

/// `nmod_poly_struct` / `nmod_poly_t[0]` view for FFI.
#[repr(C)]
pub struct NmodPolyStruct {
    pub coeffs: *mut ulong,
    pub alloc: slong,
    pub length: slong,
    pub mod_: NmodStruct,
}

/// `nmod_poly_factor_struct` / `nmod_poly_factor_t`.
#[repr(C)]
pub struct NmodPolyFactorStruct {
    pub p: *mut NmodPolyStruct,
    pub exp: *mut slong,
    pub num: slong,
    pub alloc: slong,
}

/// Multivariate factorization container (`fmpz_mpoly_factor_struct`).
/// Exponent entries are `fmpz` (FLINT stores multiplicities as small integers).
#[repr(C)]
pub struct FmpzMPolyFactorStruct {
    pub constant: fmpz,
    pub constant_den: fmpz,
    pub poly: *mut FmpzMPolyBuf,
    pub exp: *mut fmpz,
    pub num: slong,
    pub alloc: slong,
}

// SAFETY: FLINT integers and polynomials are safe to send across threads
// (they own their memory and have no thread-local state).
unsafe impl Send for FmpzPolyStruct {}
unsafe impl Sync for FmpzPolyStruct {}

/// `fmpz_mat_struct` — row-pointer layout (FLINT 2.x and FLINT 3.0.x).
#[repr(C)]
#[cfg(not(flint3_stride))]
pub struct FmpzMatStruct {
    pub entries: *mut fmpz,
    pub r: slong,
    pub c: slong,
    pub rows: *mut *mut fmpz,
}

/// `fmpz_mat_struct` — stride layout (FLINT 3.1+, detected via fmpz_mat.h).
#[repr(C)]
#[cfg(flint3_stride)]
pub struct FmpzMatStruct {
    pub entries: *mut fmpz,
    pub r: slong,
    pub c: slong,
    pub stride: slong,
}

/// `fmpz_factor_struct` / `fmpz_factor_t` — integer factorisation container.
#[repr(C)]
pub struct FmpzFactorStruct {
    pub sign: c_int,
    pub p: *mut fmpz,
    pub exp: *mut ulong,
    pub alloc: slong,
    pub num: slong,
}

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
    pub fn fmpz_cmp_ui(f: *const fmpz, x: ulong) -> c_int;
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
    pub fn fmpz_set_ui(f: *mut fmpz, val: ulong);
    pub fn fmpz_cmp(f: *const fmpz, g: *const fmpz) -> c_int;
    pub fn fmpz_cmp_si(f: *const fmpz, s: slong) -> c_int;
    pub fn fmpz_abs(f: *mut fmpz, g: *const fmpz);
    pub fn fmpz_sub_ui(f: *mut fmpz, g: *const fmpz, h: ulong);
    pub fn fmpz_mul_ui(f: *mut fmpz, g: *const fmpz, h: ulong);
    pub fn fmpz_mod(f: *mut fmpz, x: *const fmpz, m: *const fmpz);
    pub fn fmpz_powm(r: *mut fmpz, b: *const fmpz, e: *const fmpz, m: *const fmpz);
    pub fn fmpz_invmod(res: *mut fmpz, x: *const fmpz, m: *const fmpz) -> c_int;
    pub fn fmpz_sqrtmod(x: *mut fmpz, a: *const fmpz, p: *const fmpz) -> c_int;
    pub fn fmpz_jacobi(a: *const fmpz, n: *const fmpz) -> c_int;

    /// Returns `1` if \(n\) is proved prime, `0` if composite (FLINT `fmpz_is_prime`).
    pub fn fmpz_is_prime(n: *const fmpz) -> c_int;
    pub fn fmpz_nextprime(res: *mut fmpz, n: *const fmpz, proved: c_int);
    pub fn fmpz_euler_phi(res: *mut fmpz, n: *const fmpz);

    pub fn fmpz_factor_init(fac: *mut FmpzFactorStruct);
    pub fn fmpz_factor_clear(fac: *mut FmpzFactorStruct);
    pub fn fmpz_factor(fac: *mut FmpzFactorStruct, n: *const fmpz);

    pub fn fmpz_fdiv_ui(g: *const fmpz, h: ulong) -> ulong;
    pub fn fmpz_get_ui(f: *const fmpz) -> ulong;

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

    /// Complete factorization over `ℤ` (Zassenhaus / van Hoeij inside FLINT).
    pub fn fmpz_poly_factor_init(fac: *mut FmpzPolyFactorStruct);
    pub fn fmpz_poly_factor_clear(fac: *mut FmpzPolyFactorStruct);
    pub fn fmpz_poly_factor(fac: *mut FmpzPolyFactorStruct, poly: *const FmpzPolyStruct);
    pub fn fmpz_poly_factor_get_fmpz_poly(
        z: *mut FmpzPolyStruct,
        fac: *const FmpzPolyFactorStruct,
        i: slong,
    );
    pub fn fmpz_poly_factor_get_fmpz(z: *mut fmpz, fac: *const FmpzPolyFactorStruct);
    /// Swinnerton–Dyer polynomial `S_n` (test oracle / irreducibility checks).
    pub fn fmpz_poly_swinnerton_dyer(poly: *mut FmpzPolyStruct, n: ulong);
    pub fn fmpz_poly_cyclotomic(poly: *mut FmpzPolyStruct, n: ulong);

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

    // fmpz_mpoly — full factorization over `ℤ[x₁,…,xₙ]` (Bernardin–Monagan, etc.)
    pub fn fmpz_mpoly_factor_init(f: *mut FmpzMPolyFactorStruct, ctx: *const FmpzMPolyCtxBuf);
    pub fn fmpz_mpoly_factor_clear(f: *mut FmpzMPolyFactorStruct, ctx: *const FmpzMPolyCtxBuf);
    pub fn fmpz_mpoly_factor(
        f: *mut FmpzMPolyFactorStruct,
        A: *const FmpzMPolyBuf,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> c_int;
    pub fn fmpz_mpoly_factor_length(
        f: *const FmpzMPolyFactorStruct,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> slong;
    pub fn fmpz_mpoly_factor_get_base(
        p: *mut FmpzMPolyBuf,
        fac: *const FmpzMPolyFactorStruct,
        i: slong,
        ctx: *const FmpzMPolyCtxBuf,
    );
    pub fn fmpz_mpoly_factor_get_constant_fmpz(
        c: *mut fmpz,
        f: *const FmpzMPolyFactorStruct,
        ctx: *const FmpzMPolyCtxBuf,
    );
    pub fn fmpz_mpoly_factor_get_exp_si(
        f: *mut FmpzMPolyFactorStruct,
        i: slong,
        ctx: *const FmpzMPolyCtxBuf,
    ) -> slong;

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

    // nmod_poly — univariate polynomials over ℤ/pℤ (Berlekamp, Cantor–Zassenhaus, …)
    pub fn nmod_init(mod_: *mut NmodStruct, n: ulong);
    pub fn nmod_poly_init(poly: *mut NmodPolyStruct, n: ulong);
    pub fn nmod_poly_clear(poly: *mut NmodPolyStruct);
    pub fn nmod_poly_set_coeff_ui(poly: *mut NmodPolyStruct, n: slong, c: ulong);
    pub fn nmod_poly_factor_init(fac: *mut NmodPolyFactorStruct);
    pub fn nmod_poly_factor_clear(fac: *mut NmodPolyFactorStruct);
    pub fn nmod_poly_factor(
        result: *mut NmodPolyFactorStruct,
        input: *const NmodPolyStruct,
    ) -> ulong;
    /// FLINT 2.x
    #[cfg(not(flint3))]
    pub fn nmod_poly_factor_get_nmod_poly(
        z: *mut NmodPolyStruct,
        fac: *mut NmodPolyFactorStruct,
        i: slong,
    );
    /// FLINT 3.x — same role as `nmod_poly_factor_get_nmod_poly` (renamed upstream).
    #[cfg(flint3)]
    pub fn nmod_poly_factor_get_poly(
        z: *mut NmodPolyStruct,
        fac: *const NmodPolyFactorStruct,
        i: slong,
    );
    pub fn nmod_poly_degree(poly: *const NmodPolyStruct) -> slong;
    pub fn nmod_poly_get_coeff_ui(poly: *const NmodPolyStruct, j: slong) -> ulong;

    // -----------------------------------------------------------------------
    // fmpz_mat — dense integer matrices (Hermite / Smith normal forms)
    // -----------------------------------------------------------------------

    pub fn fmpz_mat_init(mat: *mut FmpzMatStruct, rows: slong, cols: slong);
    pub fn fmpz_mat_clear(mat: *mut FmpzMatStruct);
    pub fn fmpz_mat_swap(mat1: *mut FmpzMatStruct, mat2: *mut FmpzMatStruct);
    pub fn fmpz_mat_zero(mat: *mut FmpzMatStruct);
    pub fn fmpz_mat_one(mat: *mut FmpzMatStruct);
    pub fn fmpz_mat_set(dst: *mut FmpzMatStruct, src: *const FmpzMatStruct);
    pub fn fmpz_mat_equal(a: *const FmpzMatStruct, b: *const FmpzMatStruct) -> c_int;
    pub fn fmpz_mat_mul(c: *mut FmpzMatStruct, a: *const FmpzMatStruct, b: *const FmpzMatStruct);
    pub fn fmpz_mat_transpose(dst: *mut FmpzMatStruct, src: *const FmpzMatStruct);
    pub fn fmpz_mat_hnf_transform(
        h: *mut FmpzMatStruct,
        u: *mut FmpzMatStruct,
        a: *const FmpzMatStruct,
    );
    pub fn fmpz_mat_snf(s: *mut FmpzMatStruct, a: *const FmpzMatStruct);
    pub fn fmpz_mat_is_in_hnf(a: *const FmpzMatStruct) -> c_int;
    pub fn fmpz_mat_is_in_snf(a: *const FmpzMatStruct) -> c_int;
}
