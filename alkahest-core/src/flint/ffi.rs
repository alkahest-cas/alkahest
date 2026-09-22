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

/// `fmpz_lll_struct` / `fmpz_lll_t` — the LLL parameter context.
///
/// FLINT ships no `fmpz_lll.h` on this box, so this layout was recovered
/// **empirically** from FLINT 3.5.0 rather than read from a header:
///
/// * `fmpz_lll_context_init_default` writes one 16-byte SSE store at offset 0
///   (the two `double`s `delta`, `eta`) and then `movq $1, 0x10(%rdi)` — a
///   single 8-byte store covering offsets 16..24. That is two 4-byte `enum`
///   fields, `rt = 1` and `gt = 0`.
/// * `fmpz_lll_context_init` takes `delta` in `xmm0`, `eta` in `xmm1`, `rt` in
///   `esi` and `gt` in `edx`, and writes them to offsets 0, 8, 16, 20.
/// * `fmpz_lll_is_reduced` reads `cmpl $0x1, 0x10(%rsi)`, confirming `rt` is a
///   4-byte field at offset 16 whose `Z_BASIS` value is `1`.
///
/// The defaults `fmpz_lll_context_init_default` installs are `delta = 0.9925`,
/// `eta = 0.5225`, `rt = Z_BASIS`, `gt = APPROX` (read out of `.rodata`).
#[repr(C)]
pub struct FmpzLllStruct {
    pub delta: f64,
    pub eta: f64,
    /// `rep_type`: `GRAM = 0`, `Z_BASIS = 1`.
    pub rt: c_int,
    /// `gram_type`: `APPROX = 0`, `EXACT = 1`.
    pub gt: c_int,
}

/// `rep_type::Z_BASIS` — the rows of the matrix are the lattice basis.
pub const FMPZ_LLL_Z_BASIS: c_int = 1;
/// `gram_type::EXACT` — use exact (`fmpz`) Gram computations, not `double`s.
pub const FMPZ_LLL_EXACT: c_int = 1;

/// `fmpz_factor_struct` / `fmpz_factor_t` — integer factorisation container.
#[repr(C)]
pub struct FmpzFactorStruct {
    pub sign: c_int,
    pub p: *mut fmpz,
    pub exp: *mut ulong,
    pub alloc: slong,
    pub num: slong,
}

// ---------------------------------------------------------------------------
// nmod_mat / fq_nmod_mat — dense matrices over GF(p) and GF(p^k)
// ---------------------------------------------------------------------------
//
// READ BEFORE TOUCHING THESE.
//
// `nmod_mat_struct` and `fq_nmod_mat_struct` underwent the *same* layout change
// as `fmpz_mat_struct` above: FLINT 2.x / 3.0.x stored an array of row pointers,
// later FLINT 3 releases replaced it with a `stride`. Both fields are
// pointer-sized, so the two layouts have identical `size_of` and a misdetection
// is **not** a compile error and **not** a size mismatch — it is an integer
// dereferenced as a pointer, i.e. silent memory corruption.
//
// Two independent defences are in place:
//
//  1. The declarations below are selected by the same `flint3_stride` cfg that
//     `build.rs` reads out of `flint/fmpz_types.h` — the existing precedent.
//  2. **Nothing in this crate ever does pointer arithmetic on `entries`.** Every
//     entry read and write goes through FLINT's own accessors
//     (`nmod_mat_set_entry`, `nmod_mat_get_entry`, `fq_nmod_mat_entry`,
//     `fq_nmod_mat_entry_set`), which compute the offset from whichever field
//     the installed FLINT actually has. Only `r` and `c` are read directly, and
//     those sit at offsets 8 and 16 in *both* layouts.
//
// `crate::ffield::tests` additionally round-trips every entry of several
// non-square shapes through FLINT and asserts equality, which is the check that
// catches a wrong stride (a wrong stride frequently still passes on square or
// 1-column matrices).

/// `nmod_mat_struct` — row-pointer layout (FLINT 2.x and FLINT 3.0.x).
#[repr(C)]
#[cfg(not(flint3_stride))]
pub struct NmodMatStruct {
    pub entries: *mut ulong,
    pub r: slong,
    pub c: slong,
    pub rows: *mut *mut ulong,
    pub mod_: NmodStruct,
}

/// `nmod_mat_struct` — stride layout (FLINT 3.1+, detected via fmpz_types.h).
#[repr(C)]
#[cfg(flint3_stride)]
pub struct NmodMatStruct {
    pub entries: *mut ulong,
    pub r: slong,
    pub c: slong,
    pub stride: slong,
    pub mod_: NmodStruct,
}

/// `fq_nmod_mat_struct` — row-pointer layout (FLINT 2.x and FLINT 3.0.x).
///
/// Entries are `fq_nmod_struct`, which FLINT `typedef`s to `nmod_poly_struct`.
#[repr(C)]
#[cfg(not(flint3_stride))]
pub struct FqNmodMatStruct {
    pub entries: *mut NmodPolyStruct,
    pub r: slong,
    pub c: slong,
    pub rows: *mut *mut NmodPolyStruct,
}

/// `fq_nmod_mat_struct` — stride layout (FLINT 3.1+).
#[repr(C)]
#[cfg(flint3_stride)]
pub struct FqNmodMatStruct {
    pub entries: *mut NmodPolyStruct,
    pub r: slong,
    pub c: slong,
    pub stride: slong,
}

/// Opaque storage for one `fq_nmod_ctx_t`.
///
/// FLINT does not export the size and this box ships no `fq_nmod_types.h`, so
/// the struct is held as an over-sized aligned byte array rather than a mirrored
/// `#[repr(C)]` declaration — over-allocating an opaque C struct is safe, while
/// mirroring a layout that has changed twice upstream is not.
///
/// Measured on FLINT 3.5.0 (x86-64): `fq_nmod_ctx_init_ui` writes 160 bytes.
/// 512 is a 3.2× margin. `ffield::tests::fq_ctx_buffer_has_generous_slack` fills the
/// buffer with a sentinel and asserts the tail survives initialisation, so a
/// future FLINT that outgrows the buffer fails a test rather than the heap.
#[repr(C, align(8))]
pub struct FqNmodCtxBuf(pub [u8; 512]);

/// Opaque storage for one `fq_nmod_poly_t` (24 bytes on FLINT 3.5.0; 128 here).
#[repr(C, align(8))]
pub struct FqNmodPolyBuf(pub [u8; 128]);

// ---------------------------------------------------------------------------
// fmpq / fmpq_poly / nf / nf_elem — rationals, rational polynomials and the
// algebraic number fields of FLINT's absorbed Antic library
// ---------------------------------------------------------------------------
//
// READ BEFORE TOUCHING THESE.
//
// `nf_t` and `nf_elem_t` are the trap in this file. `nf_elem_struct` is a
// **union whose active member depends on the degree of the field**: Antic
// special-cases degree 1 (two `fmpz`, a numerator and a denominator) and
// degree 2 (three `fmpz` numerators and a denominator) and uses an
// `fmpq_poly_struct` for everything else. A degree-2 element therefore does
// not have the layout of a degree-3 one, and picking the wrong arm is silent
// memory corruption rather than a compile error.
//
// The defence is the same one `nmod_mat` uses above, taken further: **neither
// struct is mirrored here at all.** Both are opaque, over-sized, aligned byte
// arrays, and every read and write goes through a FLINT entry point that also
// receives the `nf_t` and so can pick the right union arm itself. Nothing in
// this crate computes an offset into either.
//
// The sizes below were measured on FLINT 3.5.0 (x86-64) with a C probe that
// filled the buffer with a sentinel byte and counted how many trailing bytes
// `nf_init` / `nf_elem_init` left untouched:
//
//   sizeof(nf_struct)      = 112  (fmpq_poly pol, preinv/powers unions,
//                                  fmpq_poly traces, ulong flag)
//   sizeof(nf_elem_struct) =  32  (degree 1 writes 16, everything else 32)
//
// `numfield::tests` re-runs that measurement from Rust — see
// `nf_buffers_have_generous_slack` — so a future FLINT that outgrows either
// buffer fails a test rather than the heap.

/// C layout of `fmpq` — FLINT's rational number, a numerator and a
/// denominator in lowest terms with a positive denominator.
///
/// Unlike `nf_elem_struct` this one *is* mirrored, because `fmpq_numref` and
/// `fmpq_denref` are C macros rather than exported symbols, so there is no
/// accessor to route through. The layout is part of FLINT's documented API and
/// was confirmed twice here: by disassembly (`arith_bernoulli_number` passes
/// `x` and `x + 8` to `_arith_bernoulli_number` as numerator and denominator)
/// and by a C probe. `numfield::tests::fmpq_layout_matches_flint` cross-checks
/// the fields against `fmpq_get_str` at run time.
#[repr(C)]
pub struct Fmpq {
    /// Numerator.
    pub num: fmpz,
    /// Denominator, always positive once canonicalised.
    pub den: fmpz,
}

/// Opaque storage for one `fmpq_poly_t` (32 bytes on FLINT 3.5.0; 64 here).
///
/// Held opaquely for the same reason as [`NfBuf`]: nothing in this crate needs
/// a field of it, and `fmpq_poly_*` covers every access.
#[repr(C, align(8))]
pub struct FmpqPolyBuf(pub [u8; 64]);

/// Opaque storage for one `nf_t` (112 bytes on FLINT 3.5.0; 512 here).
#[repr(C, align(8))]
pub struct NfBuf(pub [u8; 512]);

/// Opaque storage for one `nf_elem_t` (16 or 32 bytes depending on the degree
/// of the field; 128 here).
///
/// The size varies **with the field**, which is why this is a fixed over-sized
/// buffer and never an array: FLINT would stride an `nf_elem_struct[]` by the
/// union's size, not by this buffer's.
#[repr(C, align(8))]
pub struct NfElemBuf(pub [u8; 128]);

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
    /// Pointer to entry `(i, j)`.
    ///
    /// **Always go through this rather than doing pointer arithmetic on
    /// `entries`.** FLINT swapped `fmpz_mat_struct`'s row-pointer array for a
    /// `stride` in 3.1; both fields are pointer-sized, so guessing wrong is
    /// silent memory corruption rather than a compile error. Disassembly of
    /// FLINT 3.5.0 shows this entry point computing
    /// `entries + (mat->stride * i + j)` with `stride` at offset 24 — which is
    /// the layout the `flint3_stride` cfg selects, and is *checked at runtime*
    /// by `lattice::flint_backend::tests::entry_round_trip_non_square`.
    pub fn fmpz_mat_entry(mat: *const FmpzMatStruct, i: slong, j: slong) -> *mut fmpz;
    /// Exact determinant of a square matrix.
    pub fn fmpz_mat_det(det: *mut fmpz, a: *const FmpzMatStruct);
    pub fn fmpz_mat_rank(a: *const FmpzMatStruct) -> slong;
    /// Row-style Hermite normal form (no transform matrix).
    pub fn fmpz_mat_hnf(h: *mut FmpzMatStruct, a: *const FmpzMatStruct);

    // -----------------------------------------------------------------------
    // fmpz_lll — Lenstra–Lenstra–Lovász basis reduction
    // -----------------------------------------------------------------------
    //
    // Signatures recovered by disassembly against FLINT 3.5.0; see
    // `FmpzLllStruct` above for how the context layout was established.
    // `fmpz_lll` itself is a three-argument tail call into
    // `fmpz_lll_with_removal_ulll(B, U, 250, NULL, fl)`, so `U` may be null.
    pub fn fmpz_lll_context_init_default(fl: *mut FmpzLllStruct);
    pub fn fmpz_lll_context_init(
        fl: *mut FmpzLllStruct,
        delta: f64,
        eta: f64,
        rt: c_int,
        gt: c_int,
    );
    /// LLL-reduce the rows of `b` in place. `u`, when non-null, accumulates the
    /// unimodular transform.
    pub fn fmpz_lll(b: *mut FmpzMatStruct, u: *mut FmpzMatStruct, fl: *const FmpzLllStruct);
    /// FLINT's own reducedness oracle, to `prec` bits.
    pub fn fmpz_lll_is_reduced(
        b: *const FmpzMatStruct,
        fl: *const FmpzLllStruct,
        prec: ulong,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // Word-sized primality (used to refuse a non-prime GF(q) characteristic)
    // -----------------------------------------------------------------------
    pub fn n_is_prime(n: ulong) -> c_int;

    // -----------------------------------------------------------------------
    // nmod_poly — extra entry points for GF(p^k) defining polynomials
    // -----------------------------------------------------------------------
    pub fn nmod_poly_is_irreducible(f: *const NmodPolyStruct) -> c_int;

    // -----------------------------------------------------------------------
    // nmod_mat — dense matrices over GF(p), p prime and word-sized
    // -----------------------------------------------------------------------
    //
    // Signatures below were confirmed against FLINT 3.5.0 by disassembly
    // (which argument registers each entry point reads) rather than from a
    // header, because this box ships none. In particular `nmod_mat_rref` is
    // **in place and single-argument** — it reads only `%rdi` — while the
    // `fq_nmod_mat` sibling takes a separate destination.
    pub fn nmod_mat_init(mat: *mut NmodMatStruct, rows: slong, cols: slong, n: ulong);
    pub fn nmod_mat_clear(mat: *mut NmodMatStruct);
    pub fn nmod_mat_set(dst: *mut NmodMatStruct, src: *const NmodMatStruct);
    pub fn nmod_mat_equal(a: *const NmodMatStruct, b: *const NmodMatStruct) -> c_int;
    /// Entry write. Goes through FLINT so the row offset is computed by the
    /// installed library, never by us — see the layout note above.
    pub fn nmod_mat_set_entry(mat: *mut NmodMatStruct, i: slong, j: slong, x: ulong);
    /// Entry read. Same reasoning as `nmod_mat_set_entry`.
    pub fn nmod_mat_get_entry(mat: *const NmodMatStruct, i: slong, j: slong) -> ulong;
    pub fn nmod_mat_add(c: *mut NmodMatStruct, a: *const NmodMatStruct, b: *const NmodMatStruct);
    pub fn nmod_mat_sub(c: *mut NmodMatStruct, a: *const NmodMatStruct, b: *const NmodMatStruct);
    pub fn nmod_mat_neg(b: *mut NmodMatStruct, a: *const NmodMatStruct);
    pub fn nmod_mat_scalar_mul(b: *mut NmodMatStruct, a: *const NmodMatStruct, c: ulong);
    pub fn nmod_mat_mul(c: *mut NmodMatStruct, a: *const NmodMatStruct, b: *const NmodMatStruct);
    pub fn nmod_mat_transpose(b: *mut NmodMatStruct, a: *const NmodMatStruct);
    pub fn nmod_mat_rank(a: *const NmodMatStruct) -> slong;
    /// Reduced row echelon form, **in place**, returning the rank.
    pub fn nmod_mat_rref(a: *mut NmodMatStruct) -> slong;
    /// Right nullspace basis in the first `nullity` columns of `x`; returns the
    /// nullity. `x` must have at least `a->c` rows and `a->c` columns.
    pub fn nmod_mat_nullspace(x: *mut NmodMatStruct, a: *const NmodMatStruct) -> slong;
    /// Returns non-zero on success, `0` when `a` is singular.
    pub fn nmod_mat_inv(b: *mut NmodMatStruct, a: *const NmodMatStruct) -> c_int;
    pub fn nmod_mat_det(a: *const NmodMatStruct) -> ulong;
    pub fn nmod_mat_charpoly(p: *mut NmodPolyStruct, m: *const NmodMatStruct);
    /// Solves `a·x = b` for possibly rectangular / rank-deficient `a`.
    /// Returns `1` when a solution exists (and writes one), `0` otherwise.
    pub fn nmod_mat_can_solve(
        x: *mut NmodMatStruct,
        a: *const NmodMatStruct,
        b: *const NmodMatStruct,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // fq_nmod_ctx — GF(p^k) field context
    // -----------------------------------------------------------------------
    /// Conway polynomial when FLINT has one tabulated for `(p, d)`, otherwise a
    /// deterministic minimal-weight irreducible. Reproducible either way.
    pub fn fq_nmod_ctx_init_ui(ctx: *mut FqNmodCtxBuf, p: ulong, d: slong, var: *const c_char);
    /// Build the field from a caller-supplied irreducible `modulus` over ℤ/pℤ.
    pub fn fq_nmod_ctx_init_modulus(
        ctx: *mut FqNmodCtxBuf,
        modulus: *const NmodPolyStruct,
        var: *const c_char,
    );
    pub fn fq_nmod_ctx_clear(ctx: *mut FqNmodCtxBuf);
    pub fn fq_nmod_ctx_modulus(ctx: *const FqNmodCtxBuf) -> *const NmodPolyStruct;

    // -----------------------------------------------------------------------
    // fq_nmod — an element of GF(p^k). FLINT typedefs `fq_nmod_t` to
    // `nmod_poly_t`, so `NmodPolyStruct` is the element type, `nmod_poly_init`
    // over the characteristic is a valid `fq_nmod_init`, and the
    // `nmod_poly_*` coefficient accessors apply to it directly. Only the two
    // entry points this module actually calls are declared: every signature
    // here is exercised by `crate::ffield::tests`.
    // -----------------------------------------------------------------------
    pub fn fq_nmod_zero(rop: *mut NmodPolyStruct, ctx: *const FqNmodCtxBuf);
    pub fn fq_nmod_mul(
        rop: *mut NmodPolyStruct,
        op1: *const NmodPolyStruct,
        op2: *const NmodPolyStruct,
        ctx: *const FqNmodCtxBuf,
    );

    // -----------------------------------------------------------------------
    // fq_nmod_mat — dense matrices over GF(p^k)
    // -----------------------------------------------------------------------
    pub fn fq_nmod_mat_init(
        mat: *mut FqNmodMatStruct,
        rows: slong,
        cols: slong,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_clear(mat: *mut FqNmodMatStruct, ctx: *const FqNmodCtxBuf);
    pub fn fq_nmod_mat_set(
        dst: *mut FqNmodMatStruct,
        src: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_equal(
        a: *const FqNmodMatStruct,
        b: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    ) -> c_int;
    /// Pointer to entry `(i, j)`. Takes no context in FLINT 3.5.0 (verified by
    /// disassembly: it reads `mat`, `i`, `j` only).
    pub fn fq_nmod_mat_entry(
        mat: *const FqNmodMatStruct,
        i: slong,
        j: slong,
    ) -> *mut NmodPolyStruct;
    pub fn fq_nmod_mat_entry_set(
        mat: *mut FqNmodMatStruct,
        i: slong,
        j: slong,
        x: *const NmodPolyStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_add(
        c: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        b: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_sub(
        c: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        b: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_neg(
        b: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_mul(
        c: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        b: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_transpose(
        b: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_rank(a: *const FqNmodMatStruct, ctx: *const FqNmodCtxBuf) -> slong;
    /// Unlike `nmod_mat_rref`, this one writes to a separate destination `b`
    /// (verified by disassembly: it forwards `(b, a, gr_ctx)` to `gr_mat_rref`).
    pub fn fq_nmod_mat_rref(
        b: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    ) -> slong;
    pub fn fq_nmod_mat_nullspace(
        x: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    ) -> slong;
    pub fn fq_nmod_mat_inv(
        b: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    ) -> c_int;
    pub fn fq_nmod_mat_charpoly(
        p: *mut FqNmodPolyBuf,
        m: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    );
    pub fn fq_nmod_mat_can_solve(
        x: *mut FqNmodMatStruct,
        a: *const FqNmodMatStruct,
        b: *const FqNmodMatStruct,
        ctx: *const FqNmodCtxBuf,
    ) -> c_int;

    // -----------------------------------------------------------------------
    // fq_nmod_poly — only what `charpoly` needs to be read back
    // -----------------------------------------------------------------------
    pub fn fq_nmod_poly_init(poly: *mut FqNmodPolyBuf, ctx: *const FqNmodCtxBuf);
    pub fn fq_nmod_poly_clear(poly: *mut FqNmodPolyBuf, ctx: *const FqNmodCtxBuf);
    pub fn fq_nmod_poly_length(poly: *const FqNmodPolyBuf, ctx: *const FqNmodCtxBuf) -> slong;
    pub fn fq_nmod_poly_get_coeff(
        x: *mut NmodPolyStruct,
        poly: *const FqNmodPolyBuf,
        n: slong,
        ctx: *const FqNmodCtxBuf,
    );

    // -----------------------------------------------------------------------
    // fmpq — rationals
    // -----------------------------------------------------------------------
    pub fn fmpq_init(x: *mut Fmpq);
    pub fn fmpq_clear(x: *mut Fmpq);
    /// Reduce `num`/`den` to lowest terms with a positive denominator.
    pub fn fmpq_canonicalise(x: *mut Fmpq);
    /// Allocates and returns `"p"` or `"p/q"`. Caller must `flint_free` it.
    pub fn fmpq_get_str(str_: *mut c_char, b: c_int, x: *const Fmpq) -> *mut c_char;

    // -----------------------------------------------------------------------
    // fmpq_poly — dense univariate polynomials over ℚ
    // -----------------------------------------------------------------------
    pub fn fmpq_poly_init(poly: *mut FmpqPolyBuf);
    pub fn fmpq_poly_clear(poly: *mut FmpqPolyBuf);
    pub fn fmpq_poly_degree(poly: *const FmpqPolyBuf) -> slong;
    pub fn fmpq_poly_set_fmpz_poly(rop: *mut FmpqPolyBuf, op: *const FmpzPolyStruct);
    pub fn fmpq_poly_set_coeff_fmpq(poly: *mut FmpqPolyBuf, n: slong, x: *const Fmpq);

    // -----------------------------------------------------------------------
    // nf / nf_elem — algebraic number fields ℚ[x]/(f) (FLINT's absorbed Antic)
    // -----------------------------------------------------------------------
    //
    // Every signature below was confirmed against FLINT 3.5.0 by disassembly —
    // which argument register each entry point reads, and which one it treats
    // as the `nf_t` by loading the flag word at offset 0x68 — and then
    // exercised end to end by a C probe in degrees 1, 2, 3 and 4. This box
    // ships no `nf.h`, so there was no header to copy them from.
    //
    // Note the argument order: the `nf_t` comes **last**, and
    // `nf_elem_get_str_pretty` takes `(elem, var, nf)`.
    pub fn nf_init(nf: *mut NfBuf, pol: *const FmpqPolyBuf);
    pub fn nf_clear(nf: *mut NfBuf);

    pub fn nf_elem_init(a: *mut NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_clear(a: *mut NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_set(a: *mut NfElemBuf, b: *const NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_zero(a: *mut NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_one(a: *mut NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_gen(a: *mut NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_is_zero(a: *const NfElemBuf, nf: *const NfBuf) -> c_int;
    pub fn nf_elem_is_one(a: *const NfElemBuf, nf: *const NfBuf) -> c_int;
    pub fn nf_elem_equal(a: *const NfElemBuf, b: *const NfElemBuf, nf: *const NfBuf) -> c_int;
    pub fn nf_elem_neg(r: *mut NfElemBuf, a: *const NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_add(
        r: *mut NfElemBuf,
        a: *const NfElemBuf,
        b: *const NfElemBuf,
        nf: *const NfBuf,
    );
    pub fn nf_elem_sub(
        r: *mut NfElemBuf,
        a: *const NfElemBuf,
        b: *const NfElemBuf,
        nf: *const NfBuf,
    );
    pub fn nf_elem_mul(
        r: *mut NfElemBuf,
        a: *const NfElemBuf,
        b: *const NfElemBuf,
        nf: *const NfBuf,
    );
    pub fn nf_elem_div(
        r: *mut NfElemBuf,
        a: *const NfElemBuf,
        b: *const NfElemBuf,
        nf: *const NfBuf,
    );
    pub fn nf_elem_inv(r: *mut NfElemBuf, a: *const NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_pow(r: *mut NfElemBuf, a: *const NfElemBuf, e: ulong, nf: *const NfBuf);
    pub fn nf_elem_norm(res: *mut Fmpq, a: *const NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_trace(res: *mut Fmpq, a: *const NfElemBuf, nf: *const NfBuf);
    pub fn nf_elem_get_coeff_fmpq(c: *mut Fmpq, a: *const NfElemBuf, i: slong, nf: *const NfBuf);
    pub fn nf_elem_set_fmpq_poly(a: *mut NfElemBuf, pol: *const FmpqPolyBuf, nf: *const NfBuf);
    /// Allocates and returns the element written in terms of `var`. Caller must
    /// `flint_free` it. Argument order is `(elem, var, nf)`.
    pub fn nf_elem_get_str_pretty(
        a: *const NfElemBuf,
        var: *const c_char,
        nf: *const NfBuf,
    ) -> *mut c_char;

    // -----------------------------------------------------------------------
    // arith / bernoulli / partitions — classical arithmetic functions
    // -----------------------------------------------------------------------
    //
    // Two of these changed argument order between FLINT 2 and FLINT 3 and are
    // easy to get backwards: `fmpz_divisor_sigma` and `arith_sum_of_squares`
    // both take `(result, k, n)` here — the *exponent* before the argument.
    // Confirmed by disassembly (it is `%rdx`, not `%rsi`, that is dereferenced
    // as an `fmpz` pointer) and by a C probe against σ₁(12) = 28, σ₂(12) = 210
    // and r₂(5) = 8.
    pub fn arith_number_of_partitions(x: *mut fmpz, n: ulong);
    pub fn arith_bernoulli_number(x: *mut Fmpq, n: ulong);
    pub fn arith_euler_number(res: *mut fmpz, n: ulong);
    pub fn arith_harmonic_number(x: *mut Fmpq, n: slong);
    /// Signed Stirling number of the first kind, `s(n, k)`.
    pub fn arith_stirling_number_1(s: *mut fmpz, n: slong, k: slong);
    /// Unsigned Stirling number of the first kind, `c(n, k) = |s(n, k)|`.
    pub fn arith_stirling_number_1u(s: *mut fmpz, n: slong, k: slong);
    /// Stirling number of the second kind, `S(n, k)`.
    pub fn arith_stirling_number_2(s: *mut fmpz, n: slong, k: slong);
    /// Number of representations of `n` as an ordered sum of `k` squares.
    pub fn arith_sum_of_squares(r: *mut fmpz, k: ulong, n: *const fmpz);
    pub fn fmpz_moebius_mu(n: *const fmpz) -> c_int;
    pub fn fmpz_divisor_sigma(res: *mut fmpz, k: ulong, n: *const fmpz);

    // -----------------------------------------------------------------------
    // fmpz_poly — discriminant (used for the defining polynomial of a field)
    // -----------------------------------------------------------------------
    pub fn fmpz_poly_discriminant(res: *mut fmpz, poly: *const FmpzPolyStruct);
}
