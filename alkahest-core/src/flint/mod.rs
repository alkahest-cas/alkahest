//! Layer 0 — Foundation bindings to FLINT (Fast Library for Number Theory).
//!
//! **Design decision (v0.1):** We link against the system-installed FLINT
//! 2.8.4 (`libflint-dev`) rather than using a bundled source build.
//! Rationale: system FLINT coexists cleanly with the `rug`/`gmp-mpfr-sys`
//! dependency (no duplicate GMP symbols); build times stay short; FLINT 2.8.4
//! covers all Phase 2 requirements (`fmpz_t`, `fmpz_poly_t`).
//!
//! TODO(flint3): Before v0.5 (verified computation / IntervalValue), migrate to
//! FLINT 3.x.  FLINT 3.0 absorbed the Arb library, making `arb_t`/`acb_t` ball
//! arithmetic first-class types — required for `IntervalValue` tracers and
//! rigorous interval evaluation.  Migration scope is narrow: update `build.rs`
//! to link `flint` from `libflint3-dev`, revise the type declarations in
//! `ffi.rs` for any renamed symbols, and add `arb.rs` / `acb.rs` wrapper
//! modules.  Safe wrapper APIs above `ffi.rs` are unaffected.
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
//!
//! All raw C pointers are confined to `ffi.rs`; everything above is safe Rust.

pub(crate) mod ffi;
pub mod integer;
pub(crate) mod mat;
pub(crate) mod mpoly;
pub(crate) mod nmod;
pub mod poly;

pub use integer::FlintInteger;
pub use poly::FlintPoly;
