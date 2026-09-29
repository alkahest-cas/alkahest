//! Layer 0 — Foundation bindings to FLINT (Fast Library for Number Theory).
//!
//! **Design decision (v0.1):** We link against a system-installed FLINT
//! (`libflint-dev` / `flint-devel` / `brew install flint`, ≥ 3.3; ≥ 3.4 for
//! Riemann theta — `build.rs`'s `check_flint_min_version` enforces the
//! former) rather than
//! using a bundled source build. Rationale: system FLINT coexists cleanly with
//! the `rug`/`gmp-mpfr-sys` dependency (no duplicate GMP symbols) and build
//! times stay short.
//!
//! **This module is not optional and is not feature-gated.** It is declared
//! unconditionally from `lib.rs`; [`crate::poly::UniPoly`] stores a
//! [`FlintPoly`], and `poly::factor`, `poly::resultant`, `matrix::normal_form`,
//! `ideal::primary` and `number_theory` call FLINT directly. There is no
//! pure-Rust or MPFR-only fallback for any of it, so a build without FLINT is
//! not a supported configuration — `build.rs` refuses it up front rather than
//! producing a shared object with undefined symbols. The `flint3` feature and
//! the `flint3` / `flint3_stride` cfgs choose which FLINT *version's* ABI to
//! use; they do not make the dependency optional.
//!
//! Set `FLINT_LIB_DIR` / `FLINT_INCLUDE_DIR` to build against a FLINT in a
//! non-standard (e.g. user-local, root-free) prefix.
//!
//! The FLINT 3 migration this module once carried a TODO for is done: FLINT
//! 3.0 absorbed Arb, and `arb.rs` / `acb.rs` bind its ball types directly.
//! FLINT 2.x is no longer accepted at all (see the minimum above).
//!
//! # Memory safety design
//!
//! Every FLINT type requires a paired `*_init` / `*_clear` call.  This module
//! provides drop-safe Rust wrappers for all types used in the codebase:
//!
//! | FLINT C type            | Rust wrapper              | `Drop` calls              |
//! |-------------------------|---------------------------|---------------------------|
//! | `fmpz_t`                | [`FlintInteger`]          | `fmpz_clear`              |
//! | `fmpz_factor_t`         | `integer::FlintIntFactor`  | `fmpz_factor_clear`      |
//! | `fmpz_poly_t`           | [`FlintPoly`]             | `fmpz_poly_clear`         |
//! | `fmpz_poly_factor_t`    | `poly::FlintPolyFactor`    | `fmpz_poly_factor_clear`  |
//! | `fmpz_mpoly_ctx_t`      | `mpoly::FlintMPolyCtx`     | `fmpz_mpoly_ctx_clear`    |
//! | `fmpz_mpoly_t`          | `mpoly::FlintMPoly`        | `fmpz_mpoly_clear`        |
//! | `fmpz_mpoly_factor_t`   | `mpoly::FlintMPolyFactor`  | `fmpz_mpoly_factor_clear` |
//! | `nmod_poly_t`           | `nmod::FlintNmodPoly`      | `nmod_poly_clear`         |
//! | `nmod_poly_factor_t`    | `nmod::FlintNmodPolyFactor`| `nmod_poly_factor_clear`  |
//! | `fmpz_mat_t`            | `mat::FlintMat`            | `fmpz_mat_clear`          |
//! | `fmpq_t`                | `rational::FlintRational`  | `fmpq_clear`              |
//! | `fmpq_poly_t`           | `numfield::field::FqPoly`  | `fmpq_poly_clear`         |
//! | `nf_t` / `nf_elem_t`    | `numfield::field`          | `nf_clear` / `nf_elem_clear` |
//!
//! All raw C pointers are confined to `ffi.rs`; everything above is safe Rust.

/// Genuine `arb_t` / `acb_t` bindings (FLINT >= 3.1 only; see `build.rs`'s
/// `flint_arb` probe). Separate from, and with no effect on, the MPFR-backed
/// [`crate::ball`] module.
#[cfg(flint_arb)]
pub mod acb;
#[cfg(flint_arb)]
pub mod arb;
pub(crate) mod ffi;
pub mod integer;
pub(crate) mod mat;
pub(crate) mod mpoly;
pub(crate) mod nmod;
pub mod poly;
pub(crate) mod qgcd;
pub(crate) mod rational;

pub use integer::FlintInteger;
pub use poly::FlintPoly;

/// The factor containers' `*_at(i)` accessors are safe fns over raw FLINT
/// arrays. Their bounds check used to be a `debug_assert!`, so an index past
/// `len()` was an out-of-bounds read in a release build (UB) and a panic only
/// in debug. These pin the panic in both (`cargo test --release` included).
#[cfg(test)]
mod factor_index_tests {
    use super::integer::{FlintInteger, FlintIntFactor};
    use super::mpoly::{FlintMPoly, FlintMPolyCtx, FlintMPolyFactor};
    use super::nmod::{FlintNmodPoly, FlintNmodPolyFactor};
    use super::poly::{FlintPoly, FlintPolyFactor};

    fn int_factor_of_12() -> FlintIntFactor {
        let mut f = FlintIntFactor::new();
        f.factor(&FlintInteger::from_i64(12));
        assert_eq!(f.len(), 2);
        f
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn int_factor_base_at_past_len() {
        let f = int_factor_of_12();
        let _ = f.base_at(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn int_factor_exp_at_past_len() {
        let f = int_factor_of_12();
        let _ = f.exp_at(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn int_factor_exp_at_on_empty() {
        let _ = FlintIntFactor::new().exp_at(0);
    }

    fn poly_factor_of_x2_minus_1() -> FlintPolyFactor {
        let mut f = FlintPolyFactor::new();
        f.factor(&FlintPoly::from_coefficients(&[-1, 0, 1]));
        assert_eq!(f.len(), 2);
        f
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn poly_factor_poly_at_past_len() {
        let _ = poly_factor_of_x2_minus_1().poly_at(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn poly_factor_exp_at_past_len() {
        let _ = poly_factor_of_x2_minus_1().exp_at(2);
    }

    fn nmod_factor_of_x2_minus_1() -> FlintNmodPolyFactor {
        let mut p = FlintNmodPoly::new(7);
        p.set_coeff(0, 6);
        p.set_coeff(2, 1);
        let mut f = FlintNmodPolyFactor::new();
        f.factor(&p);
        assert_eq!(f.len(), 2);
        f
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn nmod_factor_exp_at_past_len() {
        let _ = nmod_factor_of_x2_minus_1().exp_at(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn nmod_factor_poly_at_past_len() {
        let _ = nmod_factor_of_x2_minus_1().poly_at(7, 2);
    }

    fn mpoly_factor_of_x2_minus_y2() -> FlintMPolyFactor {
        let ctx = FlintMPolyCtx::new(2);
        let mut p = FlintMPoly::new(ctx.clone());
        p.push_term(&rug::Integer::from(1), &[2, 0]);
        p.push_term(&rug::Integer::from(-1), &[0, 2]);
        p.finish();
        let mut f = FlintMPolyFactor::new(ctx);
        assert!(f.factor(&p));
        assert_eq!(f.len(), 2);
        f
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn mpoly_factor_base_at_past_len() {
        let _ = mpoly_factor_of_x2_minus_y2().base_at(2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn mpoly_factor_exp_at_past_len() {
        let _ = mpoly_factor_of_x2_minus_y2().exp_at(2);
    }
}
