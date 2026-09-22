//! [`LatticeGeometryError`] — the refusals of the lattice *toolkit*, as opposed
//! to those of LLL reduction.
//!
//! # Why this is a second enum
//!
//! [`LatticeError`] covers basis reduction and predates this module. It is a
//! public, **exhaustive** enum re-exported from `alkahest_cas::stable`, so
//! adding a variant to it is a semver-major change — `cargo semver-checks`
//! reports `enum_variant_added` and demands a new major version. Forcing a
//! major bump for a new experimental subsystem is the wrong trade, so the
//! toolkit's refusals live here instead.
//!
//! This enum is **`#[non_exhaustive]` from birth**, which is the point: every
//! future lattice-toolkit refusal can be added without another break. The two
//! can be consolidated at a deliberate major bump later; until then the split
//! is a versioning artefact, not a semantic one, and both sides share the
//! `E-LAT-NNN` code space. The Python bindings make
//! `LatticeGeometryError` a **subclass of `LatticeError`**, so
//! `except LatticeError` keeps catching everything the subsystem raises.
//!
//! # Codes
//!
//! `E-LAT-001` … `E-LAT-004` stay with [`LatticeError`] and reach a caller
//! through [`LatticeGeometryError::Reduction`], which delegates its code,
//! message and remediation to the error it wraps. `E-LAT-005` … `E-LAT-014`
//! are this enum's own.

use super::lll::LatticeError;
use crate::errors::AlkahestError;
use std::fmt;

/// Refusals from the lattice toolkit: Gram matrices, positive-definiteness,
/// enumeration limits, and the constructors.
///
/// `#[non_exhaustive]`: match with a `_` arm. New variants will be added.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatticeGeometryError {
    /// A basis reduction failed. Carries `E-LAT-001` … `E-LAT-004` unchanged.
    ///
    /// Present because the toolkit reduces bases: shape validation
    /// (`EmptyBasis`, `RaggedBasis`) and the exact LLL iteration guard reach a
    /// caller of [`super::Lattice`] exactly as they reach a caller of
    /// [`super::lattice_reduce_rows`], and re-coding them here would have
    /// given the same condition two codes.
    Reduction(LatticeError),
    /// A Gram matrix must be square.
    NonSquareGram { rows: usize, cols: usize },
    /// A Gram matrix must be symmetric; `G[row][col] != G[col][row]`.
    AsymmetricGram { row: usize, col: usize },
    /// The quadratic form is not positive definite — the `pivot`-th leading
    /// principal minor is not positive, so the "lattice" is degenerate.
    NotPositiveDefinite { pivot: usize },
    /// Enumeration (SVP/CVP/theta) was asked for above the hard rank ceiling.
    RankTooLarge { rank: usize, max: usize },
    /// Enumeration exhausted its node budget without finishing.
    EnumerationBudget { budget: u64 },
    /// An integral Gram matrix is required (theta series, kissing numbers);
    /// entry `(row, col)` is not an integer.
    NonIntegralGram { row: usize, col: usize },
    /// A supplied vector has the wrong length for this lattice.
    DimensionMismatch { expected: usize, got: usize },
    /// A constructor parameter is out of the supported range.
    InvalidParameter { detail: &'static str },
    /// The operation needs ambient coordinates and the lattice was built from a
    /// Gram matrix alone.
    NoBasis,
    /// An internal invariant did not hold.
    ///
    /// This exists so that a "cannot happen" branch is a **typed refusal rather
    /// than a panic**: these functions are called across a PyO3 boundary, where
    /// a panic arrives as `pyo3_runtime.PanicException` — a `BaseException`
    /// that a caller's `except Exception` does not catch.
    Internal { detail: &'static str },
}

impl fmt::Display for LatticeGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LatticeGeometryError::Reduction(e) => write!(f, "{e}"),
            LatticeGeometryError::NonSquareGram { rows, cols } => {
                write!(f, "a Gram matrix must be square; got {rows}x{cols}")
            }
            LatticeGeometryError::AsymmetricGram { row, col } => write!(
                f,
                "Gram matrix is not symmetric: entry ({row},{col}) differs from ({col},{row})"
            ),
            LatticeGeometryError::NotPositiveDefinite { pivot } => write!(
                f,
                "quadratic form is not positive definite (leading minor {pivot} is not positive)"
            ),
            LatticeGeometryError::RankTooLarge { rank, max } => write!(
                f,
                "lattice enumeration is capped at rank {max}; this lattice has rank {rank}"
            ),
            LatticeGeometryError::EnumerationBudget { budget } => write!(
                f,
                "lattice enumeration exceeded its budget of {budget} nodes without finishing"
            ),
            LatticeGeometryError::NonIntegralGram { row, col } => write!(
                f,
                "this operation needs an integral Gram matrix; entry ({row},{col}) is not an integer"
            ),
            LatticeGeometryError::DimensionMismatch { expected, got } => {
                write!(f, "expected a vector of length {expected}, got {got}")
            }
            LatticeGeometryError::InvalidParameter { detail } => {
                write!(f, "unsupported lattice parameter: {detail}")
            }
            LatticeGeometryError::NoBasis => write!(
                f,
                "this lattice was built from a Gram matrix and has no ambient coordinates"
            ),
            LatticeGeometryError::Internal { detail } => {
                write!(f, "internal lattice invariant violated: {detail}")
            }
        }
    }
}

impl std::error::Error for LatticeGeometryError {}

impl From<LatticeError> for LatticeGeometryError {
    fn from(e: LatticeError) -> Self {
        LatticeGeometryError::Reduction(e)
    }
}

impl AlkahestError for LatticeGeometryError {
    fn code(&self) -> &'static str {
        match self {
            LatticeGeometryError::Reduction(e) => e.code(),
            LatticeGeometryError::NonSquareGram { .. } => "E-LAT-005",
            LatticeGeometryError::AsymmetricGram { .. } => "E-LAT-006",
            LatticeGeometryError::NotPositiveDefinite { .. } => "E-LAT-007",
            LatticeGeometryError::RankTooLarge { .. } => "E-LAT-008",
            LatticeGeometryError::EnumerationBudget { .. } => "E-LAT-009",
            LatticeGeometryError::NonIntegralGram { .. } => "E-LAT-010",
            LatticeGeometryError::DimensionMismatch { .. } => "E-LAT-011",
            LatticeGeometryError::InvalidParameter { .. } => "E-LAT-012",
            LatticeGeometryError::NoBasis => "E-LAT-013",
            LatticeGeometryError::Internal { .. } => "E-LAT-014",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            LatticeGeometryError::Reduction(e) => e.remediation(),
            LatticeGeometryError::NonSquareGram { .. } => {
                Some("a Gram matrix has one row and one column per basis vector")
            }
            LatticeGeometryError::AsymmetricGram { .. } => {
                Some("a Gram matrix is G[i][j] = <b_i, b_j>; supply the full symmetric matrix")
            }
            LatticeGeometryError::NotPositiveDefinite { .. } => Some(
                "basis rows must be linearly independent and a Gram matrix positive definite",
            ),
            LatticeGeometryError::RankTooLarge { .. } => Some(
                "enumeration is exponential in the rank; project to a sublattice or use LLL/BKZ approximations instead",
            ),
            LatticeGeometryError::EnumerationBudget { .. } => Some(
                "raise the node budget explicitly, lower the norm bound, or reduce the basis first",
            ),
            LatticeGeometryError::NonIntegralGram { .. } => Some(
                "scale the lattice so that all inner products are integers, or ask for the minimum instead of a theta series",
            ),
            LatticeGeometryError::DimensionMismatch { .. } => {
                Some("supply a vector with one entry per ambient coordinate")
            }
            LatticeGeometryError::InvalidParameter { .. } => {
                Some("check the documented parameter range for this constructor")
            }
            LatticeGeometryError::NoBasis => Some(
                "build the lattice from a basis (Lattice::from_basis) if you need ambient coordinates",
            ),
            LatticeGeometryError::Internal { .. } => Some(
                "this is a bug: please report it with the lattice that produced it",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wrapped reduction errors must keep their own codes rather than
    /// acquiring a new one — otherwise the split would have given one condition
    /// two identities.
    #[test]
    fn wrapped_reduction_errors_delegate_their_code() {
        let inner = LatticeError::RaggedBasis {
            row: 1,
            expected_cols: 3,
            got_cols: 2,
        };
        let wrapped: LatticeGeometryError = inner.clone().into();
        assert_eq!(wrapped.code(), "E-LAT-002");
        assert_eq!(wrapped.code(), inner.code());
        assert_eq!(wrapped.remediation(), inner.remediation());
        assert_eq!(wrapped.to_string(), inner.to_string());
    }

    #[test]
    fn every_variant_has_a_distinct_code_in_the_expected_range() {
        let all = [
            LatticeGeometryError::NonSquareGram { rows: 1, cols: 2 },
            LatticeGeometryError::AsymmetricGram { row: 0, col: 1 },
            LatticeGeometryError::NotPositiveDefinite { pivot: 1 },
            LatticeGeometryError::RankTooLarge { rank: 25, max: 24 },
            LatticeGeometryError::EnumerationBudget { budget: 1 },
            LatticeGeometryError::NonIntegralGram { row: 0, col: 0 },
            LatticeGeometryError::DimensionMismatch {
                expected: 2,
                got: 1,
            },
            LatticeGeometryError::InvalidParameter { detail: "x" },
            LatticeGeometryError::NoBasis,
            LatticeGeometryError::Internal { detail: "x" },
        ];
        let codes: Vec<&str> = all.iter().map(|e| e.code()).collect();
        assert_eq!(
            codes,
            [
                "E-LAT-005",
                "E-LAT-006",
                "E-LAT-007",
                "E-LAT-008",
                "E-LAT-009",
                "E-LAT-010",
                "E-LAT-011",
                "E-LAT-012",
                "E-LAT-013",
                "E-LAT-014",
            ]
        );
        for e in &all {
            assert!(e.remediation().is_some(), "{e} has no remediation");
        }
    }
}
