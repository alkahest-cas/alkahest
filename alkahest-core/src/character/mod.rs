//! Conjugacy classes and **exact** ordinary character tables of finite
//! permutation groups.
//!
//! ```
//! use alkahest_cas::experimental::{alternating, CharacterTable};
//!
//! // A_4: four classes, degrees 1, 1, 1, 3 — and two of the linear characters
//! // are genuinely irrational, valued in the cube roots of unity.
//! let table = CharacterTable::of(&alternating(4).unwrap()).unwrap();
//! assert_eq!(table.degrees(), &[1, 1, 1, 3]);
//! assert_eq!(table.classes().len(), 4);
//! assert_eq!(table.exponent(), 6);            // so values live in Q(zeta_6)
//!
//! // Row 0 is the trivial character; rows 1 and 2 are the two conjugate
//! // linear characters, which take non-rational values.
//! assert!(table.character(1).unwrap().iter().any(|v| v.coefficients()[1] != 0));
//! ```
//!
//! # What is here
//!
//! * [`ConjugacyClasses`] — the class partition of a [`crate::group::PermutationGroup`]:
//!   representatives (the lexicographically smallest element of each class),
//!   class sizes, centraliser orders, the class of any given element, `exp G`,
//!   the inverse-class involution, and the class multiplication coefficients
//!   `a_{ijk}` both entrywise and as the matrices `M_k` Dixon's algorithm
//!   diagonalises.
//! * [`ConjugacyClass`] — one class.
//! * [`CharacterTable`] — the irreducible characters by **Dixon–Schneider**,
//!   with values as exact elements of the cyclotomic field `ℚ(ζ_{exp G})`,
//!   their degrees, the table indexed by class, the weighted inner product, and
//!   a `Display` that lays it out as a grid.
//!
//! # Conventions inherited from [`crate::group`]
//!
//! **Points are 0-based** and **composition is left-to-right** (`p.compose(&q)`
//! applies `p` first). Products in the class algebra are formed with `compose`,
//! so `a_{ijk}` counts pairs `(x, y)` with `x` applied first.
//!
//! Nothing this module *returns* depends on that choice, which is worth saying
//! rather than leaving to be guessed: a set of generators generates the same
//! subgroup either way, conjugacy is conjugacy, and the class algebra is
//! **commutative**, so `a_{ijk} = a_{jik}` and the coefficients come out the
//! same under the opposite convention too. (`multiplication_coefficient` is
//! tested for that symmetry.) What the convention fixes is the *definition* —
//! which of `xy` and `yx` the count is over — so that it is unambiguous, not
//! the number.
//!
//! # Exactness, and the ceilings
//!
//! Every value is an element of one cyclotomic field; nothing is floating point
//! and nothing degrades to rationals when an irrationality appears. `A_4`'s
//! cube roots of unity and `A_5`'s golden-ratio pair `(1 ± √5)/2` are handled
//! by the same code path as `S_4`'s integers. See [`table`] for the algorithm
//! and for why the lift out of GF(p) is exact rather than heuristic.
//!
//! Two ceilings, both enforced rather than documented:
//!
//! * **`|G|`.** Classes are found by conjugating every element, so the whole
//!   group is held in memory as one images array per element:
//!   [`DEFAULT_CLASS_ENUMERATION_CAP`] for [`ConjugacyClasses::of`],
//!   [`DEFAULT_CHARACTER_TABLE_CAP`] for [`CharacterTable::of`], and
//!   [`MAX_CLASS_ENUMERATION_CAP`] as the hard limit on either. Past it the
//!   answer is [`CharacterError::OrderTooLarge`] naming the cap — never a
//!   partial list of classes, which would be a wrong answer rather than a
//!   coarse one. Nothing here uses a randomised or backtrack-search class
//!   algorithm, which is what a real system needs past `10^6`.
//! * **`φ(exp G)`.** Every value lives in `ℚ(ζ_{exp G})` and the orthogonality
//!   checks multiply there `O(r³)` times — which is also what makes the number
//!   of *classes* the practical limit rather than the order: `S_7`
//!   (`|G| = 5040`, `r = 15`) takes under a tenth of a second, `ℤ/100`
//!   (`r = 100`) about fifteen. The field degree is capped at
//!   [`MAX_EXPONENT_FIELD_DEGREE`] ([`CharacterError::ExponentFieldTooLarge`]).
//!   The two ceilings are independent: a group of small exponent passes this
//!   one at any order (and meets the `|G|` cap instead), while the cyclic group
//!   of order 1000 is refused here despite being small.
//!
//! # Scope limits worth stating plainly
//!
//! * **Non-abelian irrationalities are handled in general, not case by case.**
//!   The Dixon lift reconstructs each value as a sum of roots of unity with
//!   integer multiplicities; there is no table of special cases, no
//!   `√5`-detector, and no rational fallback. `A_4` and `A_5` are tested
//!   because they are the smallest groups where a rationals-only
//!   implementation would be caught, not because they are special-cased.
//! * **Nothing unchecked is returned.** Row and column orthogonality,
//!   `Σ_i χ_i(1)² = |G|`, `χ_i(1) | |G|`, and "one character per class" are all
//!   asserted as exact identities in `ℚ(ζ_e)` before a table exists. A
//!   violation is [`CharacterError::SelfCheckFailed`] and the table is
//!   withheld.
//! * **Not implemented:** Brauer characters and modular representation theory;
//!   the power map, Galois action and Schur indices (beyond the inverse-class
//!   involution that `conj` needs); induced, restricted, symmetric and exterior
//!   powers; tensor decomposition into irreducibles; tables of finitely
//!   presented or matrix groups; and any table read out of a library instead of
//!   computed. `ConjugacyClasses` also does *not* give centralisers as
//!   subgroups — only their orders, which come free from orbit–stabilizer.

pub mod classes;
mod dixon;
pub mod error;
pub mod table;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;

pub use classes::{
    ConjugacyClass, ConjugacyClasses, DEFAULT_CLASS_ENUMERATION_CAP, MAX_CLASS_ENUMERATION_CAP,
};
pub use error::CharacterError;
pub use table::{CharacterTable, DEFAULT_CHARACTER_TABLE_CAP, MAX_EXPONENT_FIELD_DEGREE};
