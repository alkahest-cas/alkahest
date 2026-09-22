//! Classical linear codes over GF(q), weight enumerators, and the Delsarte
//! linear-programming bound.
//!
//! ```
//! use alkahest_cas::experimental::{delsarte_lp_bound, FiniteField, LinearCode};
//! use rug::Integer;
//!
//! // The [7,4,3] Hamming code and its weight distribution.
//! let gf2 = FiniteField::prime(2).unwrap();
//! let hamming = LinearCode::hamming(&gf2, 3).unwrap();
//! assert_eq!((hamming.length(), hamming.dimension()), (7, 4));
//! assert_eq!(hamming.minimum_distance().unwrap(), Some(3));
//!
//! let w = hamming.weight_enumerator().unwrap();
//! let a: Vec<i64> = w.coefficients().iter().map(|c| c.to_i64().unwrap()).collect();
//! assert_eq!(a, [1, 0, 0, 7, 7, 0, 0, 1]);
//!
//! // MacWilliams sends it to the [7,3,4] simplex code, which is its dual.
//! let dual = w.macwilliams().unwrap();
//! let b: Vec<i64> = dual.coefficients().iter().map(|c| c.to_i64().unwrap()).collect();
//! assert_eq!(b, [1, 0, 0, 0, 7, 0, 0, 0]);
//!
//! // And the LP says no code of length 7 and distance 3 can beat it.
//! let lp = delsarte_lp_bound(7, 3, 2).unwrap();
//! assert_eq!(*lp.bound(), Integer::from(16));
//! ```
//!
//! # What is here
//!
//! * [`LinearCode`] — an `[n, k]` subspace of `GF(q)^n`, carried as a generator
//!   **and** a parity-check matrix, each derived from the other through
//!   [`crate::ffield::GfMatrix`]. Duals, extension by an overall parity symbol,
//!   self-duality, membership, and the Hamming, Golay and repetition families.
//! * [`WeightEnumerator`] — exact `rug::Integer` weight distributions, the
//!   enumerator polynomial, and the **MacWilliams transform**, which is
//!   verified in both directions against directly enumerated duals in the
//!   tests.
//! * [`krawtchouk`] / [`krawtchouk_poly`] — the eigenvalues of the Hamming
//!   association scheme, exactly.
//! * [`delsarte_lp_bound`] — the headline: an upper bound on `A_q(n, d)`
//!   computed by an exact-rational simplex and returned **with the dual
//!   certificate that proves it**. See [`delsarte`] for why that distinction
//!   matters.
//! * [`singleton_bound`] and [`hamming_bound`] — elementary cross-checks the LP
//!   bound must never exceed.
//!
//! # What is deliberately not here
//!
//! * **A clever minimum-distance algorithm.** Minimum distance is computed by
//!   enumerating all `q^k` codewords, under a hard cap
//!   ([`MAX_ENUMERATED_CODEWORDS`]) with a typed refusal above it. There is no
//!   Brouwer–Zimmermann. The problem is NP-hard, the published speed-ups are
//!   intricate, and a subtly wrong `d` reported confidently is worse than an
//!   honest `E-CODE-004`.
//! * **Decoding.** No syndrome tables, no Berlekamp–Massey, no list decoding.
//!   [`LinearCode::contains`] tells you whether a word is a codeword; it will
//!   not tell you which codeword a corrupted word came from.
//! * **Cyclic-code machinery.** [`LinearCode::golay_binary`] happens to be
//!   built from a generator polynomial, but there is no `CyclicCode` type, no
//!   BCH bound, no Reed–Solomon or Reed–Muller constructor, and no
//!   automorphism group.
//! * **Quantum / stabiliser codes.** A different subsystem; the GF(4) and
//!   symplectic machinery they need is not here.
//! * **Non-linear codes as objects.** The Delsarte bound applies to arbitrary
//!   codes — that is the point of it — but the only code *object* in this
//!   module is a linear subspace.
//! * **The strengthened LP bounds.** Schrijver's semidefinite programme and the
//!   extra inequalities that beat plain Delsarte on specific `(n, d)` are not
//!   implemented. What is returned is always a valid upper bound; it is not
//!   always the best one known.
//!
//! # Error codes
//!
//! `E-CODE-001` … `E-CODE-008`, all on [`CodingError`]; see
//! [`crate::errors::codes::REGISTRY`].

pub mod code;
pub mod delsarte;
pub mod error;
pub mod krawtchouk;
pub mod weight;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

pub use code::{LinearCode, MAX_ENUMERATED_CODEWORDS, MAX_ENUMERATION_CELLS};
pub use delsarte::{
    delsarte_lp_bound, hamming_bound, singleton_bound, DelsarteBound, MAX_LP_LENGTH,
};
pub use error::CodingError;
pub use krawtchouk::{
    binomial, binomial_generalised, krawtchouk, krawtchouk_pairing, krawtchouk_poly,
};
pub use weight::WeightEnumerator;
