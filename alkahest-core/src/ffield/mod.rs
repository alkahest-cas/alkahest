//! Linear algebra over finite fields GF(q), `q = p^k`.
//!
//! This module exists to serve **linear codes over GF(q)** — classical and
//! quantum error-correcting codes, and the cryptography / zero-knowledge uses
//! that share the same primitives. The operation those workloads execute most
//! is the nullspace of a rectangular parity-check matrix over GF(2), so that is
//! the path this module is built and tested around: nothing here assumes a
//! square matrix, and `rank + nullity == ncols` is asserted by both a unit test
//! and a property test.
//!
//! ```
//! use alkahest_cas::experimental::{FiniteField, GfMatrix};
//!
//! // The [7,4] Hamming code's parity-check matrix, over GF(2).
//! let gf2 = FiniteField::prime(2).unwrap();
//! let h = GfMatrix::from_u64(&gf2, 3, 7, &[
//!     1, 0, 1, 0, 1, 0, 1,
//!     0, 1, 1, 0, 0, 1, 1,
//!     0, 0, 0, 1, 1, 1, 1,
//! ]).unwrap();
//!
//! assert_eq!(h.rank(), 3);
//! let g_t = h.nullspace().unwrap();          // 7 x 4: columns span ker H
//! assert_eq!(g_t.ncols(), 4);                // dimension 4 => a [7,4] code
//! assert!(h.mul(&g_t).unwrap().is_zero());   // H · Gᵀ = 0
//! ```
//!
//! # What is here
//!
//! * [`FiniteField`] — GF(p) for a word-sized prime `p`, and GF(p^k) built
//!   either from FLINT's Conway polynomial or from a caller-supplied
//!   irreducible polynomial (checked for irreducibility before use).
//! * [`GfMatrix`] — dense matrices, of any shape: `add`, `sub`, `neg`,
//!   `scalar_mul`, `mul`, `transpose`, `rank`, [`GfMatrix::rref`] (with the
//!   transform `U` such that `U·A = R`), [`GfMatrix::nullspace`],
//!   [`GfMatrix::solve`], [`GfMatrix::inverse`], [`GfMatrix::determinant`] and
//!   [`GfMatrix::charpoly`].
//!
//! Everything is backed by FLINT: `nmod_mat_*` for prime fields, `fq_nmod_mat_*`
//! for extensions.
//!
//! # What is deliberately *not* here
//!
//! * **Composite moduli.** ℤ/nℤ for composite `n` is a ring with zero divisors.
//!   Gaussian elimination in it can meet a pivot that is non-zero and not
//!   invertible, so "rank" and "nullspace" stop being well defined. A non-prime
//!   characteristic is refused (`E-GFQ-001`) rather than approximated. FLINT's
//!   `nmod_mat_howell_form` is the right tool for that case and is not wrapped.
//! * **Multi-precision characteristics.** `nmod`/`fq_nmod` are word-sized by
//!   construction; a prime beyond `u64` is refused (`E-GFQ-002`). The FLINT type
//!   for that is `fmpz_mod_mat`, which this module does not wrap.
//! * **Sparse and bit-packed representations.** Every matrix here is dense, and
//!   GF(2) entries occupy a full 64-bit word each rather than one bit. A
//!   bit-packed GF(2) kernel would be several times faster on the large sparse
//!   parity-check matrices that LDPC and surface-code work uses; FLINT's
//!   `nmod_mat` is what is wrapped today, and its asymptotics (`O(n³)` word
//!   operations) are what you get.
//! * **Field extensions of extensions, embeddings, and the Frobenius map.**
//!   Elements carry coordinates in *one* fixed polynomial basis. GF(2³) built
//!   from `x³+x+1` and GF(2³) built from `x³+x²+1` are isomorphic and are
//!   treated here as different fields (`E-GFQ-006`), because an element's
//!   coordinates mean different things in each and silently identifying them
//!   would be the wrong answer in the shape this crate refuses by policy.
//! * **Symbolic entries.** These matrices hold field elements, not
//!   [`crate::kernel::ExprId`]s. For symbolic linear algebra use
//!   [`crate::matrix`].
//! * **Eigenvalues, Jordan / rational canonical form, minimal polynomial.**
//!   `charpoly` is provided; factoring it over GF(q) is
//!   [`crate::poly::factor`]'s job and is not wired up here.
//!
//! # A note on the FFI
//!
//! `nmod_mat_struct` and `fq_nmod_mat_struct` changed layout in FLINT 3.1 —
//! row pointers became a stride — in exactly the way `fmpz_mat_struct` did.
//! Both layouts have the same size, so getting it wrong is silent memory
//! corruption rather than a compile error. The declarations in
//! `crate::flint::ffi` follow the existing `flint3_stride` cfg precedent,
//! **and** nothing in this module ever computes an entry address itself: every
//! read and write goes through FLINT's own accessors. The round-trip tests in
//! `ffield::tests` fill several non-square shapes entry by entry and read them
//! all back, which is the check a wrong stride fails (and which a square-only
//! test would pass).

pub mod error;
pub mod field;
pub mod matrix;

#[cfg(test)]
mod tests;

pub use error::FiniteFieldError;
pub use field::{FieldElement, FiniteField, MAX_EXTENSION_DEGREE};
pub use matrix::{GfMatrix, Rref};
