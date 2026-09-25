//! Matrix groups over GF(q): **real algorithms on arbitrary generators**, not
//! order formulas.
//!
//! [`MatGroup`] is a subgroup of `GL(d, q)` given by a list of invertible
//! matrices — any list, not only the classical families. From the generators
//! alone this module computes
//!
//! * the exact order and membership, from a base and strong generating set built
//!   by Schreier–Sims on the action on vectors ([`MatrixStabilizerChain`]);
//! * orbits on vectors and on **projective points** (1-dimensional subspaces),
//!   the latter being what keeps degrees manageable for `PGL`/`PSL`;
//! * pseudo-random elements by product replacement, seeded deterministically;
//! * the derived subgroup, the centre, and the normal closure of a subset;
//! * the induced permutation actions, handed to [`crate::group::PermutationGroup`].
//!
//! ```
//! use alkahest_cas::experimental::{FiniteField, MatGroup};
//!
//! let gf5 = FiniteField::prime(5).unwrap();
//! let sl = MatGroup::special_linear(&gf5, 2).unwrap();
//!
//! // |SL(2,5)| = 120, from Schreier-Sims on the generators.
//! assert_eq!(sl.order().unwrap(), 120);
//! // SL(2,5) is perfect: it is the double cover of A5.
//! assert_eq!(sl.derived_subgroup().unwrap().order().unwrap(), 120);
//! // Its centre is {+-I}.
//! assert_eq!(sl.centre().unwrap().order().unwrap(), 2);
//! ```
//!
//! # The convention to read before anything else
//!
//! **The action is on row vectors, `v ↦ v·M`.** So a product `A·B` means "apply
//! `A`, then `B`", which is the same left-to-right order as
//! [`crate::group::Permutation::compose`], and the permutation representations
//! this module hands to [`crate::group`] compose the same way round as the
//! matrices they came from. It is the *opposite* of the column-vector
//! convention `v ↦ M·v` used in most linear-algebra texts. Mixing the two gives
//! a group of the right order whose elements are the transposes/inverses of the
//! ones intended.
//!
//! A pleasant consequence: `eᵢ·M` is row `i` of `M`. That is why the stabilizer
//! chain's base points are always standard basis vectors — a matrix fixing every
//! `eᵢ` is the identity, so a base inside the standard basis always exists, the
//! chain has at most `d` levels, and the innermost step of sifting is a row read
//! rather than a matrix–vector product.
//!
//! # Relationship to [`crate::stabilizer::MatrixGroup`]
//!
//! [`MatGroup`] **supersedes** [`crate::stabilizer::MatrixGroup`] and does not
//! extend it. The older type is a *descriptor* of one of three classical
//! families: it answers `order()` from a product formula, `contains()` from the
//! defining condition, and `elements()` by brute force over all `q^{d²}`
//! matrices. It has no generators and no stabilizer chain, so it cannot answer
//! anything about a subgroup someone hands it.
//!
//! Extending it in place was not available: its `MatrixGroupKind` is a public
//! exhaustive enum re-exported from `alkahest_cas::experimental`, so the
//! `Generic` variant a generator-based group needs would be a semver-major
//! change (`cargo semver-checks` reports `enum_variant_added`). The old type is
//! therefore left exactly as it is, and nothing here calls into it. Where both
//! can answer the same question — the order of `GL`, `SL`, `Sp` — they are
//! asserted to agree, which makes the formulas an oracle for the Schreier–Sims
//! and vice versa.
//!
//! # Scope limits, enforced rather than documented
//!
//! A matrix group over GF(q) can be astronomically large, so every entry point
//! that could search without bound refuses instead. The ceilings:
//!
//! * **Degree** at most [`MAX_MATGROUP_DEGREE`]. The algorithms store explicit
//!   transversals and are written for small degrees; `GL(50, 5)` needs the
//!   Aschbacher-class / constructive-recognition machinery, which is not here.
//! * **Basic orbits** at most [`MAX_MATGROUP_ORBIT`] points, each costing a
//!   transversal matrix and its inverse — `O(Σᵢ |Δᵢ| · d²)` words. Because an
//!   orbit on vectors is bounded by `q^d − 1`, this is in practice a cap on
//!   `q^d` and **not** on `|G|`: `|Sp(12, 3)| ≈ 10^{40}` is fine, while
//!   `GL(2, 4096)` is refused.
//! * **Schreier–Sims work** at most [`MAX_MATGROUP_SCHREIER_WORK`] elementary
//!   matrix operations. Exhausting it is
//!   [`MatGroupError::WorkBudgetExhausted`], never a partial chain: an
//!   incomplete chain reports an order that is a proper *divisor* of the true
//!   one, which is the confident wrong answer this crate exists to refuse.
//!   [`MatGroup::with_budget`] raises it deliberately.
//! * **Element enumeration** at most [`DEFAULT_MATGROUP_ELEMENT_CAP`]. The
//!   order stays exact above it; it is the list that is refused.
//! * **GF(q) enumeration** at most [`MAX_MATGROUP_FIELD_ORDER`]. Order,
//!   membership and orbits of a *given* vector never enumerate the field; the
//!   operations that do are the projective point lists, the classical
//!   constructors and the centre.
//!
//! # What is deliberately not here
//!
//! * **`GU`, `SU`, `SO`, `O`, `Ω` and the twisted types.** See
//!   [`classical`]'s docs: each needs its own canonical form (several, for the
//!   orthogonal groups in characteristic 2) and a generating set correct in
//!   every degenerate case. `MatGroup::new` builds any of them from generators
//!   the caller supplies, and everything generic here then applies.
//! * **Projective groups as groups.** `PGL`/`PSL`/`PSp` are available as the
//!   *permutation* group of the projective action
//!   ([`MatGroup::permutation_action_on_projective_points`]), not as a
//!   `MatGroup` — a quotient of matrix groups is not a matrix group, and
//!   inventing a faithful matrix representation of it is a different task.
//! * **Randomised (Monte-Carlo) Schreier–Sims and the near-linear-time
//!   algorithms.** The chain here is the deterministic Schreier-generator
//!   algorithm: always correct, and fast enough at the degrees this module
//!   admits.
//! * **Aschbacher class recognition, composition series, Sylow subgroups,
//!   conjugacy classes, centralizers of arbitrary elements, subgroup
//!   intersections.** Everything needing backtrack search is absent;
//!   `derived_subgroup`, `centre` and `normal_closure` are here precisely
//!   because each has a route that avoids it — normal closure by conjugating
//!   generators to a fixed point, and the centre by a *linear* computation (see
//!   [`MatGroup::commutant_basis`]).
//! * **Matrices over rings other than GF(q).** Integral and `p`-adic matrix
//!   groups are a different subject.
//!
//! # Error codes
//!
//! [`MatGroupError`] owns `E-MATGRP-001` … `E-MATGRP-013` and is
//! `#[non_exhaustive]`. It carries `E-GFQ-*` and `E-GRP-*` through delegating
//! variants that keep the wrapped error's own code — see [`error`].

pub mod chain;
pub mod classical;
mod element;
pub mod error;
pub mod group;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;

pub use chain::{
    MatrixOrbit, MatrixSchreierEntry, MatrixSiftResult, MatrixStabilizerChain,
    MatrixStabilizerLevel,
};
pub use classical::{gl_order, sl_order, sp_order};
pub use error::MatGroupError;
pub use group::MatGroup;

/// The largest matrix size this module accepts.
///
/// The chain stores explicit `d × d` transversal matrices, and every orbit is
/// inside `GF(q)^d`, so the cost grows fast in `d`. Sixteen covers `GL(16, 2)`,
/// `Sp(16, 2)`, `GL(6, 4)` and everything smaller, which is the regime the
/// deterministic algorithms here are appropriate for.
pub const MAX_MATGROUP_DEGREE: usize = 16;

/// The largest basic orbit, in points.
///
/// Each point costs a transversal matrix **and** its inverse, so a level with
/// `|Δ|` points costs `2·|Δ|·d²` words. Since an orbit on vectors is bounded by
/// `q^d − 1`, this is effectively a cap on `q^d`: `4096` admits `GL(12, 2)`,
/// `GL(6, 4)`, `GL(4, 8)`, `GL(4, 5)` and `Sp(8, 3)`, and refuses
/// `GL(2, 10007)`.
pub const MAX_MATGROUP_ORBIT: usize = 4096;

/// The default Schreier–Sims work budget, in elementary matrix operations.
///
/// Hitting it is [`MatGroupError::WorkBudgetExhausted`] — a refusal, not a
/// partial chain. [`MatGroup::with_budget`] raises it.
pub const MAX_MATGROUP_SCHREIER_WORK: u64 = 50_000_000;

/// The default cap on [`MatGroup::elements`].
///
/// The order is exact above it; the *list* is what is refused.
pub const DEFAULT_MATGROUP_ELEMENT_CAP: u64 = 100_000;

/// The largest `q` for the operations that enumerate GF(q) itself.
///
/// Order, membership and the orbit of a given vector never do; the projective
/// point list, the classical constructors and the centre do.
pub const MAX_MATGROUP_FIELD_ORDER: u64 = 4096;

/// The largest centralizing algebra [`MatGroup::centre_elements`] will
/// enumerate, in elements (`q^dim`).
pub const MAX_MATGROUP_COMMUTANT_ELEMENTS: u64 = 65_536;
