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
//! All raw C pointers are confined to `ffi.rs`; `integer.rs` and `poly.rs`
//! expose only safe Rust APIs.

pub(crate) mod ffi;
pub mod integer;
pub(crate) mod mat;
pub(crate) mod mpoly;
pub mod poly;

pub use integer::FlintInteger;
pub use poly::FlintPoly;
