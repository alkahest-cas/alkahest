//! [`MatGroupError`] — the refusals of the matrix-group layer.
//!
//! # Why a new enum
//!
//! Every enum re-exported from `alkahest_cas::stable` is exhaustive and public,
//! so adding a variant to one is a semver-major change that `cargo
//! semver-checks` reports as `enum_variant_added`. The matrix-group layer sits
//! on top of two existing subsystems — [`FiniteFieldError`] (`E-GFQ-*`) for the
//! GF(q) linear algebra and [`GroupError`] (`E-GRP-*`) for the permutation
//! actions — and rather than widen either, it carries them through **delegating
//! variants** that forward `code()`, `remediation()` and `Display` to the error
//! they wrap. A caller who sees `E-GFQ-009` (singular matrix) underneath a
//! matrix-group call gets the code the GF(q) layer raised, not a relabelled one.
//!
//! This follows the precedent set by
//! [`LatticeGeometryError::Reduction`](crate::lattice::LatticeGeometryError) and
//! [`ArithmeticError::Input`](crate::number_theory::arith::ArithmeticError).
//!
//! This enum is `#[non_exhaustive]` from birth so that later refusals do not
//! need another break.
//!
//! # Codes
//!
//! `E-MATGRP-001` … `E-MATGRP-013` are this enum's own. `E-GFQ-*` and
//! `E-GRP-*` reach a caller unchanged through
//! [`MatGroupError::FiniteField`] and [`MatGroupError::Permutation`].

use crate::errors::AlkahestError;
use crate::ffield::FiniteFieldError;
use crate::group::GroupError;
use std::fmt;

/// Refusals from the matrix-group layer.
///
/// `#[non_exhaustive]`: match with a `_` arm. New variants will be added.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MatGroupError {
    /// A GF(q) linear-algebra operation refused. Carries its `E-GFQ-*` code
    /// unchanged.
    FiniteField(FiniteFieldError),
    /// A permutation-group operation refused — reached only from the induced
    /// actions. Carries its `E-GRP-*` code unchanged.
    Permutation(GroupError),
    /// A group element has to be square.
    NotSquare {
        /// Rows of the offending matrix.
        rows: usize,
        /// Columns of the offending matrix.
        cols: usize,
    },
    /// A matrix or vector does not have the group's degree.
    ///
    /// Degrees are never padded: `GL(2, q)` and `GL(3, q)` act on different
    /// spaces and which one a caller meant is not recoverable from the
    /// argument.
    DegreeMismatch {
        /// The degree this group acts in.
        expected: usize,
        /// The degree that was supplied.
        got: usize,
    },
    /// A matrix is over a different finite field.
    ///
    /// GF(2³) built from `x³+x+1` and GF(2³) built from `x³+x²+1` are
    /// isomorphic and are *different fields* here, because an element's
    /// coordinates mean different things in each — see [`crate::ffield`].
    FieldMismatch {
        /// The group's field.
        expected: String,
        /// The supplied matrix's field.
        got: String,
    },
    /// A proposed generator is singular, so it generates no group.
    SingularGenerator {
        /// Position in the supplied generator list.
        index: usize,
    },
    /// The degree exceeds [`MAX_MATGROUP_DEGREE`](super::MAX_MATGROUP_DEGREE).
    DegreeTooLarge {
        /// The degree asked for.
        degree: usize,
        /// The largest degree this implementation accepts.
        max: usize,
    },
    /// An orbit outgrew [`MAX_MATGROUP_ORBIT`](super::MAX_MATGROUP_ORBIT).
    ///
    /// Each stabilizer-chain level stores one transversal matrix (and its
    /// inverse) per orbit point, so the memory is `O(Σ_i |Δ_i| · d²)`.
    OrbitTooLarge {
        /// How many points had been found when the cap was hit.
        points: usize,
        /// The cap in force.
        max: usize,
    },
    /// Schreier–Sims ran past its work budget without finishing.
    ///
    /// This is a **refusal, not a partial chain**: an incomplete chain reports
    /// a group order that is a proper divisor of the true one, which is exactly
    /// the confident wrong answer this crate refuses to produce.
    WorkBudgetExhausted {
        /// The budget in force, in elementary matrix operations.
        budget: u64,
    },
    /// The group is too large to enumerate element by element.
    ///
    /// The order is still exact; it is the *list* that is refused.
    EnumerationTooLarge {
        /// Decimal rendering of `|G|` (it may not fit a `u64`).
        order: String,
        /// The cap in force.
        cap: u64,
    },
    /// An operation needs to enumerate GF(q) itself and `q` is past
    /// [`MAX_MATGROUP_FIELD_ORDER`](super::MAX_MATGROUP_FIELD_ORDER).
    FieldTooLarge {
        /// Decimal rendering of `q`.
        order: String,
        /// The cap in force.
        max: u64,
    },
    /// A constructor was asked for a group it does not build.
    UnsupportedConstruction {
        /// Which constructor: `"GL"`, `"Sp"`, `"Singer"`, …
        family: &'static str,
        /// Why this `(n, q)` has no construction here.
        reason: &'static str,
    },
    /// The centralizing algebra is too large to enumerate, so the centre
    /// cannot be read off it.
    CommutantTooLarge {
        /// `dim_{GF(q)}` of the algebra of matrices commuting with every
        /// generator.
        dimension: usize,
        /// Decimal rendering of `q^dimension`.
        elements: String,
        /// The cap in force.
        cap: u64,
    },
    /// The zero vector has no projective point and no canonical scaling.
    ZeroVector,
    /// An internal invariant did not hold.
    ///
    /// A typed refusal rather than a panic: these functions run under a PyO3
    /// boundary, where a panic arrives as `pyo3_runtime.PanicException`, a
    /// `BaseException` that a caller's `except Exception` does not catch.
    Internal {
        /// What failed, concretely.
        detail: &'static str,
    },
}

impl fmt::Display for MatGroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatGroupError::FiniteField(e) => write!(f, "{e}"),
            MatGroupError::Permutation(e) => write!(f, "{e}"),
            MatGroupError::NotSquare { rows, cols } => write!(
                f,
                "a matrix-group element must be square; got {rows}x{cols}"
            ),
            MatGroupError::DegreeMismatch { expected, got } => write!(
                f,
                "this group acts in dimension {expected}; got something of dimension {got}. \
                 Degrees are not padded automatically"
            ),
            MatGroupError::FieldMismatch { expected, got } => write!(
                f,
                "this group is over {expected} and the matrix is over {got}; two isomorphic \
                 fields with different defining polynomials are different fields here"
            ),
            MatGroupError::SingularGenerator { index } => write!(
                f,
                "generator {index} is singular, so it is not an invertible matrix and \
                 generates no group"
            ),
            MatGroupError::DegreeTooLarge { degree, max } => {
                write!(f, "degree {degree} exceeds the matrix-group limit of {max}")
            }
            MatGroupError::OrbitTooLarge { points, max } => write!(
                f,
                "an orbit on vectors passed {points} points, above the cap of {max}; the \
                 stabilizer chain stores a transversal matrix per orbit point"
            ),
            MatGroupError::WorkBudgetExhausted { budget } => write!(
                f,
                "Schreier–Sims exhausted its budget of {budget} matrix operations without \
                 completing the chain; no order is reported, because an incomplete chain \
                 reports a proper divisor of the true order"
            ),
            MatGroupError::EnumerationTooLarge { order, cap } => write!(
                f,
                "the group has order {order}, above the enumeration cap of {cap}; its order \
                 is known exactly but its elements will not be listed"
            ),
            MatGroupError::FieldTooLarge { order, max } => write!(
                f,
                "this operation enumerates GF(q) itself and q = {order} is above the cap of \
                 {max}"
            ),
            MatGroupError::UnsupportedConstruction { family, reason } => {
                write!(f, "no {family} construction here: {reason}")
            }
            MatGroupError::CommutantTooLarge {
                dimension,
                elements,
                cap,
            } => write!(
                f,
                "the algebra commuting with the generators has GF(q)-dimension {dimension}, \
                 so enumerating it means {elements} matrices, above the cap of {cap}"
            ),
            MatGroupError::ZeroVector => write!(
                f,
                "the zero vector spans no 1-dimensional subspace and has no canonical scaling"
            ),
            MatGroupError::Internal { detail } => {
                write!(f, "internal matrix-group invariant violated: {detail}")
            }
        }
    }
}

impl std::error::Error for MatGroupError {}

impl From<FiniteFieldError> for MatGroupError {
    fn from(e: FiniteFieldError) -> Self {
        MatGroupError::FiniteField(e)
    }
}

impl From<GroupError> for MatGroupError {
    fn from(e: GroupError) -> Self {
        MatGroupError::Permutation(e)
    }
}

impl AlkahestError for MatGroupError {
    fn code(&self) -> &'static str {
        match self {
            MatGroupError::FiniteField(e) => e.code(),
            MatGroupError::Permutation(e) => e.code(),
            MatGroupError::NotSquare { .. } => "E-MATGRP-001",
            MatGroupError::DegreeMismatch { .. } => "E-MATGRP-002",
            MatGroupError::FieldMismatch { .. } => "E-MATGRP-003",
            MatGroupError::SingularGenerator { .. } => "E-MATGRP-004",
            MatGroupError::DegreeTooLarge { .. } => "E-MATGRP-005",
            MatGroupError::OrbitTooLarge { .. } => "E-MATGRP-006",
            MatGroupError::WorkBudgetExhausted { .. } => "E-MATGRP-007",
            MatGroupError::EnumerationTooLarge { .. } => "E-MATGRP-008",
            MatGroupError::FieldTooLarge { .. } => "E-MATGRP-009",
            MatGroupError::UnsupportedConstruction { .. } => "E-MATGRP-010",
            MatGroupError::CommutantTooLarge { .. } => "E-MATGRP-011",
            MatGroupError::ZeroVector => "E-MATGRP-012",
            MatGroupError::Internal { .. } => "E-MATGRP-013",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            MatGroupError::FiniteField(e) => e.remediation(),
            MatGroupError::Permutation(e) => e.remediation(),
            MatGroupError::NotSquare { .. } => {
                Some("group elements are d x d matrices; pass a square matrix")
            }
            MatGroupError::DegreeMismatch { .. } => Some(
                "build the matrix at the group's degree; vectors are 1 x d row vectors, since \
                 the action here is v -> v*M",
            ),
            MatGroupError::FieldMismatch { .. } => Some(
                "build the matrix over the same FiniteField object the group was built with, \
                 or rebuild the group over the matrix's field",
            ),
            MatGroupError::SingularGenerator { .. } => {
                Some("check the determinant of every generator before building the group")
            }
            MatGroupError::DegreeTooLarge { .. } => Some(
                "the matrix-group algorithms here store explicit transversals and are built \
                 for small degrees; GL(d, q) for large d needs the Aschbacher/constructive \
                 recognition machinery, which is not implemented",
            ),
            MatGroupError::OrbitTooLarge { .. } => Some(
                "an orbit on vectors is bounded by q^d - 1, so this is a cap on q^d rather \
                 than on |G|; use a smaller field or degree, or work with the projective \
                 action, whose orbits are (q^d - 1)/(q - 1) times smaller",
            ),
            MatGroupError::WorkBudgetExhausted { .. } => Some(
                "raise the budget with `order_with_budget` / `stabilizer_chain_with_budget` \
                 if the group really is this large, or supply a smaller generating set",
            ),
            MatGroupError::EnumerationTooLarge { .. } => Some(
                "use `order()`, `contains()` or `random_element()` instead of listing the \
                 elements, or raise the cap with `elements_with_cap`",
            ),
            MatGroupError::FieldTooLarge { .. } => Some(
                "orbits, order and membership do not enumerate GF(q); the operations that do \
                 (projective point lists, classical constructors, the centre) need a small q",
            ),
            MatGroupError::UnsupportedConstruction { .. } => Some(
                "build the group from explicit generators with `MatGroup::new`; the classical \
                 constructors here cover GL, SL and Sp only",
            ),
            MatGroupError::CommutantTooLarge { .. } => Some(
                "a large centralizing algebra means the module is very reducible; split it \
                 first, or compute the centre of a smaller group",
            ),
            MatGroupError::ZeroVector => {
                Some("pass a non-zero row vector; the zero vector is fixed by every matrix")
            }
            MatGroupError::Internal { .. } => {
                Some("this is a bug: please report it with the generators that produced it")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrapped GF(q) and permutation errors keep their own codes; the split is
    /// a versioning artefact and must not give one condition two identities.
    #[test]
    fn wrapped_errors_delegate_their_code() {
        let inner = FiniteFieldError::Singular { shape: (2, 2) };
        let wrapped: MatGroupError = inner.clone().into();
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.remediation(), inner.remediation());
        assert_eq!(wrapped.to_string(), inner.to_string());

        let inner = GroupError::DegreeMismatch { left: 3, right: 4 };
        let wrapped: MatGroupError = inner.clone().into();
        assert_eq!(wrapped.code(), "E-GRP-002");
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.to_string(), inner.to_string());
    }

    #[test]
    fn every_variant_has_a_distinct_code_and_remediation() {
        let all = [
            MatGroupError::NotSquare { rows: 2, cols: 3 },
            MatGroupError::DegreeMismatch {
                expected: 3,
                got: 2,
            },
            MatGroupError::FieldMismatch {
                expected: "GF(2)".into(),
                got: "GF(3)".into(),
            },
            MatGroupError::SingularGenerator { index: 0 },
            MatGroupError::DegreeTooLarge {
                degree: 99,
                max: 16,
            },
            MatGroupError::OrbitTooLarge {
                points: 5000,
                max: 4096,
            },
            MatGroupError::WorkBudgetExhausted { budget: 1 },
            MatGroupError::EnumerationTooLarge {
                order: "10".into(),
                cap: 1,
            },
            MatGroupError::FieldTooLarge {
                order: "10".into(),
                max: 1,
            },
            MatGroupError::UnsupportedConstruction {
                family: "GU",
                reason: "x",
            },
            MatGroupError::CommutantTooLarge {
                dimension: 9,
                elements: "512".into(),
                cap: 1,
            },
            MatGroupError::ZeroVector,
            MatGroupError::Internal { detail: "x" },
        ];
        let codes: Vec<&str> = all.iter().map(|e| e.code()).collect();
        assert_eq!(
            codes,
            [
                "E-MATGRP-001",
                "E-MATGRP-002",
                "E-MATGRP-003",
                "E-MATGRP-004",
                "E-MATGRP-005",
                "E-MATGRP-006",
                "E-MATGRP-007",
                "E-MATGRP-008",
                "E-MATGRP-009",
                "E-MATGRP-010",
                "E-MATGRP-011",
                "E-MATGRP-012",
                "E-MATGRP-013",
            ]
        );
        for e in &all {
            assert!(e.remediation().is_some(), "{e} has no remediation");
            assert!(!e.to_string().is_empty());
        }
    }
}
