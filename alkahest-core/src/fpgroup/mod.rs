//! Computational group theory: **finitely presented groups**.
//!
//! A presentation `⟨X | R⟩` is a finite alphabet together with a finite list of
//! words that are declared trivial. This module builds them ([`FpGroup`]),
//! enumerates the cosets of a subgroup by **Todd–Coxeter**
//! ([`CosetTable`]), hands the resulting action to the permutation-group
//! machinery ([`CosetTable::permutation_group`]), computes the abelianisation
//! `G/[G, G]` ([`FpGroup::abelian_invariants`]), rewrites a finite-index
//! subgroup's presentation by **Reidemeister–Schreier**
//! ([`SubgroupPresentation`]), and computes `H⁰`, `H¹` and `H²` of a finite
//! group with coefficients in a finitely generated abelian module
//! ([`FpGroup::cohomology`]).
//!
//! # The one thing to understand before using this
//!
//! **Almost everything here is undecidable in general, and the error type says
//! which questions were answered and which were abandoned.** There is no
//! algorithm that decides whether `⟨X | R⟩` is finite, whether two words are
//! equal, or whether a subgroup has finite index. Coset enumeration is the tool
//! anyway, because it *terminates on the cases that matter* — but it can also
//! run forever, so it runs under a cap and refuses.
//!
//! Two refusals exist and they are **not** interchangeable:
//!
//! | code | meaning |
//! |---|---|
//! | `E-FPGRP-004` [`FpGroupError::EnumerationIncomplete`] | the enumeration hit its coset cap. **Says nothing about whether the group is finite.** |
//! | `E-FPGRP-005` [`FpGroupError::ProvablyInfinite`] | the group **is** infinite — its abelianisation has an infinite cyclic factor, which is a terminating computation. |
//!
//! `⟨a, b | a², b³, (ab)⁷⟩` is the `(2,3,7)` triangle group: infinite, with a
//! *trivial* abelianisation. It therefore lands on `E-FPGRP-004`, and that is
//! the honest answer — this module cannot prove it infinite and does not
//! pretend to. `⟨a, b | [a,b]⟩` is `ℤ²`, whose abelianisation is `ℤ²`, so
//! `order()` on it lands on `E-FPGRP-005` without enumerating anything.
//! Conflating the two would let a caller read "I gave up" as "no such order
//! exists", which is the worst failure mode a CAS has.
//!
//! # Conventions
//!
//! * **Letters are signed and 1-based.** `+k` is the `k`-th generator, `-k` its
//!   inverse, and `0` is not a letter; the identity is the empty [`Word`]. Every
//!   `Word` is freely reduced on construction.
//! * **Cosets are 0-based, and coset `0` is the subgroup `H`.** This matches the
//!   0-based points of [`crate::group`], so
//!   [`CosetTable::permutation_group`] needs no renumbering.
//! * The coset action is a **right** action (`c ↦ c·g`), and
//!   [`crate::group`]'s composition is left-to-right, so tracing a word through
//!   the table and composing the corresponding permutations agree letter for
//!   letter.
//!
//! # What is reused rather than rebuilt
//!
//! * [`crate::group::PermutationGroup`] — the coset table *is* a permutation
//!   representation, so orbits, Schreier–Sims, the exact order and membership
//!   come from there. For `H = 1` that gives an independent recomputation of
//!   `|G|` through code this module did not write, which is how the `A₅` test
//!   knows the enumeration is right and not merely self-consistent.
//! * [`crate::matrix::normal_form::smith_invariants`] and
//!   [`crate::matrix::normal_form::hermite_form`] — the abelianisation and both
//!   cohomology groups are integer normal forms. There is no second Smith
//!   implementation here.
//!
//! # Scope limits
//!
//! * **Enumeration strategy**: HLT with lookahead and table compaction. Felsch,
//!   and the standardisation/canonicalisation of coset tables, are not
//!   implemented. See [`todd_coxeter`] for why HLT was the right default here.
//! * **No simplification of presentations.** Reidemeister–Schreier returns the
//!   raw rewritten presentation; there is no Tietze pass, so a cyclic subgroup
//!   of index 30 arrives with 31 generators.
//! * **Cohomology is degrees 0–2, small groups, small modules.** `|G| ≤ 32`,
//!   `rank(M) ≤ 8`, and `rank(M)·|G|^(d+1) ≤ 2000`, which puts `H²` at
//!   `|G| ≤ 12` for a rank-1 module — `A₄` is inside it and `S₄` is not.
//!   Refusing past that is deliberate: see [`cohomology`].
//! * **No `KBMAG`-style rewriting systems, no automatic structure, no
//!   low-index-subgroup search, no Tietze/Nielsen simplification, no
//!   polycyclic presentations.** None of those are approximated anywhere.
//! * The subgroup of a coset enumeration is given by **generator words**;
//!   subgroup *cosets* of subgroups given any other way are not supported.

mod abelian;
pub mod cohomology;
mod error;
mod presentation;
mod reidemeister;
pub mod todd_coxeter;
mod word;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

pub use abelian::AbelianInvariants;
pub use cohomology::{
    GModule, MAX_COCHAIN_DIMENSION, MAX_COHOMOLOGY_DEGREE, MAX_COHOMOLOGY_GROUP_ORDER,
    MAX_MODULE_RANK,
};
pub use error::FpGroupError;
pub use presentation::FpGroup;
pub use reidemeister::{reidemeister_schreier, SubgroupPresentation};
pub use todd_coxeter::{
    default_max_cosets, enumerate, CosetTable, DEFAULT_MAX_COSETS, MAX_COSET_TABLE_CELLS,
};
pub use word::{FreeGroup, Word, MAX_FREE_RANK};
