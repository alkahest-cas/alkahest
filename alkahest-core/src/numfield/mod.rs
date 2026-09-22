//! Algebraic number fields `ℚ[x]/(f)`, on FLINT's `nf` / `nf_elem` (the
//! absorbed Antic library).
//!
//! ```
//! use alkahest_cas::experimental::NumberField;
//! use rug::Rational;
//!
//! // Q(sqrt 2) = Q[x]/(x^2 - 2), coefficients ascending.
//! let k = NumberField::from_integer_coeffs(&[(-2).into(), 0.into(), 1.into()]).unwrap();
//! let sqrt2 = k.generator();
//! let one = k.one();
//! let u = one.add(&sqrt2).unwrap();              // the fundamental unit 1 + sqrt 2
//!
//! assert_eq!(u.norm(), Rational::from(-1));
//! assert_eq!(u.trace(), Rational::from(2));
//! assert_eq!(u.to_string(), "a + 1");
//!
//! // Cyclotomic fields are a first-class constructor.
//! let c = NumberField::cyclotomic(5).unwrap();
//! assert_eq!(c.degree(), 4);                      // phi(5)
//! let z = c.generator();
//! assert_eq!(z.norm(), Rational::from(1));
//! assert_eq!(z.trace(), Rational::from(-1));
//! ```
//!
//! # What is here
//!
//! * [`NumberField`] — `ℚ[x]/(f)` for an `f ∈ ℚ[x]` that is **checked for
//!   irreducibility**, never assumed to be irreducible; the degree, the
//!   canonical defining polynomial, and the discriminant *of that polynomial*.
//! * [`NumberField::cyclotomic`] — `ℚ(ζ_n)`, of degree `φ(n)`, together with
//!   the free function [`cyclotomic_polynomial`] for `Φ_n` on its own.
//! * [`NumberFieldElement`] — `add`, `sub`, `mul`, `div`, `inverse`, `pow`,
//!   equality, `norm`, `trace`, [`NumberFieldElement::minimal_polynomial`],
//!   and a `Display` that writes the element in terms of the generator.
//!
//! # Scope limits — read this before trusting a discriminant
//!
//! **The field discriminant `d_K` and the ring of integers `O_K` are not
//! computed.** [`NumberField::polynomial_discriminant`] returns
//! `disc(f)`, the discriminant of the defining polynomial, and those are
//! different numbers: for a monic integral `f` with root `a`,
//!
//! ```text
//!     disc(f) = [O_K : Z[a]]^2 · d_K
//! ```
//!
//! so they agree exactly when `ℤ[a]` is already the maximal order. `ℚ[x]/(x²−5)`
//! has `disc(f) = 20` and `d_K = 5`, because the golden-ratio-like algebraic
//! integer `(1+√5)/2` lies outside `ℤ[√5]`. Reporting one as the other is a
//! classic error and this module does not make it: the accessor is named for
//! what it computes, and there is no `field_discriminant` to mistake it for.
//! Getting `d_K` needs a maximal-order algorithm (Round 2 / Zassenhaus, or
//! Pohst–Zassenhaus with a factored `disc(f)`), which is not implemented here.
//!
//! Also deliberately absent:
//!
//! * **Ideals, class groups, unit groups, `S`-units.** No ideal arithmetic, no
//!   class number, no regulator, no Minkowski bound. Antic does not provide
//!   them either — that is Pari/Hecke territory.
//! * **Embeddings into `ℝ` and `ℂ`, signatures, complex conjugation.**
//!   Everything here is exact and algebraic; nothing is ever evaluated
//!   numerically, so a "real" field and a complex one behave identically.
//! * **Relative extensions, subfields, Galois groups, isomorphism testing.**
//!   A field is `ℚ[x]/(f)` for one absolute `f`. Two fields with different
//!   canonical defining polynomials are treated as different even when they
//!   are isomorphic — `ℚ[x]/(x²−2)` and `ℚ[x]/(x²−8)` are both `ℚ(√2)`, and
//!   mixing their elements raises `E-NUMF-006`, because the coordinates of an
//!   element mean different things in each. This is the same policy
//!   [`crate::ffield`] applies to GF(p^k).
//! * **Factoring polynomials over `K`, or root-finding in `K`.**
//! * **Integral bases, prime decomposition, valuations.**
//!
//! # A note on the FFI
//!
//! `nf_elem_t` is a C **union over the degree of the field**: Antic
//! special-cases degree 1 (a numerator and a denominator) and degree 2 (three
//! numerators and a denominator) and uses an `fmpq_poly_struct` for everything
//! else. A degree-2 element therefore does not have the layout of a degree-3
//! one, and choosing the wrong arm would be silent memory corruption rather
//! than a compile error.
//!
//! Nothing in this module knows the layout. Both `nf_t` and `nf_elem_t` are
//! opaque over-sized byte buffers, and every read and write goes through a
//! FLINT entry point that also receives the `nf_t` and so selects the arm
//! itself — the same discipline [`crate::ffield`] applies to `nmod_mat`'s
//! stride. Every signature was established by disassembling FLINT 3.5.0 (this
//! box ships no `nf.h`) and then exercised by a C probe. `numfield::tests`
//! covers degrees 1, 2, 3 and 4 explicitly, which is what makes the union
//! visible, and re-measures the buffer slack so a future FLINT that outgrows
//! it fails a test rather than the heap.

pub mod error;
pub mod field;

#[cfg(test)]
mod tests;

pub use error::NumberFieldError;
pub use field::{cyclotomic_polynomial, NumberField, NumberFieldElement, MAX_FIELD_DEGREE};
