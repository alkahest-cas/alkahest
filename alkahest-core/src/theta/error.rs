//! Refusals from the theta / modular-function surface.
//!
//! The rule this module is built around: **a ball whose radius swamps its
//! midpoint is not an answer.** Every entry point either returns an enclosure
//! the caller can interrogate, or one of these — never a plausible-looking
//! `f64` with nothing behind it.

use std::fmt;

/// Why a theta, modular or elliptic evaluation refused.
///
/// Codes are `E-THETA-001` … `E-THETA-012`; see
/// [`crate::errors::codes::REGISTRY`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ThetaError {
    /// The FLINT this crate was built against does not carry the Arb/Acb
    /// layer, or does not carry the FLINT 3.2+ `acb_theta` interface.
    ///
    /// Arb became part of FLINT in 3.0 and `acb_theta` was rewritten in 3.2;
    /// `build.rs` probes `libflint`'s symbol table and sets `flint_arb` /
    /// `flint_acb_theta` accordingly. A build against FLINT 2.x — which the
    /// manylinux wheel job still falls back to — compiles this module but
    /// cannot evaluate anything in it.
    BackendUnavailable {
        /// Which capability was missing: `"arb"` or `"acb_theta"`.
        capability: &'static str,
    },
    /// The installed FLINT's `arb_struct` / `acb_struct` layout is not the one
    /// this build was compiled against.
    ///
    /// Detected at run time by differencing two `acb_mat_entry_ptr` results
    /// and by reading back a ball set to an exactly representable value. This
    /// is deliberately a refusal and not an assertion: the alternative to
    /// noticing is silent memory corruption.
    AbiMismatch {
        /// What was measured, naming the offending number.
        detail: String,
    },
    /// The requested working precision is outside `2 ..= MAX_PRECISION_BITS`.
    PrecisionOutOfRange {
        /// Bits requested.
        bits: u64,
        /// The largest this module accepts.
        max: u32,
    },
    /// The genus is outside `1 ..= MAX_GENUS`.
    ///
    /// `acb_theta_all` returns `4^g` values, so the output alone is
    /// exponential in `g` before any work is done. The cap is a refusal, not a
    /// silent truncation.
    GenusOutOfRange {
        /// Genus asked for.
        genus: usize,
        /// The largest this module accepts.
        max: usize,
    },
    /// A vector or matrix had the wrong length for the genus.
    DimensionMismatch {
        /// What was being built, e.g. `"period matrix"`.
        what: &'static str,
        /// Entries expected.
        expected: usize,
        /// Entries supplied.
        got: usize,
    },
    /// The period matrix is not symmetric.
    ///
    /// Symmetry is checked as **ball identity** (`acb_equal`), not as overlap:
    /// two different enclosures of the same number are two different inputs,
    /// and choosing one of them for the caller would be a guess. Build the
    /// matrix with [`crate::theta::SiegelMatrix::from_upper_triangle`] if the
    /// entries were computed separately.
    NotSymmetric {
        /// Row of the offending pair.
        i: usize,
        /// Column of the offending pair.
        j: usize,
    },
    /// `Im(tau)` could not be **proved** positive definite at the working
    /// precision, so `tau` is not certainly in the Siegel upper half-space
    /// `H_g` and theta does not converge there.
    ///
    /// A refusal here means "not proved", never "proved false": raising the
    /// precision may settle it.
    NotInSiegelUpperHalfSpace {
        /// Genus of the matrix that was offered.
        genus: usize,
        /// Precision at which positive-definiteness was attempted.
        prec: u32,
    },
    /// The characteristic index `ab` is outside `0 .. 4^g`.
    CharacteristicOutOfRange {
        /// The index offered.
        ab: u64,
        /// Genus.
        genus: usize,
    },
    /// `Im(tau)` could not be proved positive, so `tau` is not certainly in the
    /// upper half-plane, where the genus-1 modular functions live.
    NotInUpperHalfPlane {
        /// Which function was asked for, e.g. `"j"`.
        function: &'static str,
    },
    /// A value was computed, and then **withheld** because its enclosure was
    /// not accurate enough to be worth reading.
    ///
    /// Raised only by the [`crate::theta::Precision::AccurateTo`] mode, after
    /// the working precision has been raised as far as this module will go.
    /// The midpoint is not returned: a midpoint with no bits behind it is the
    /// confident-wrong-answer shape this whole module exists to avoid.
    InsufficientAccuracy {
        /// Which function was asked for.
        function: &'static str,
        /// Bits of relative accuracy the caller asked for.
        requested_bits: u32,
        /// Bits of relative accuracy actually achieved, or `None` when the
        /// result came back with an infinite radius.
        achieved_bits: Option<i64>,
        /// Highest working precision that was tried.
        at_precision: u32,
    },
    /// The result came back with an infinite radius: FLINT could not bound it
    /// at all at this precision.
    ///
    /// Typically the argument is outside the domain — `℘` at a lattice point,
    /// a `tau` whose imaginary part is indistinguishable from zero — or the
    /// input ball was already indeterminate.
    Indeterminate {
        /// Which function was asked for.
        function: &'static str,
    },
    /// A midpoint or radius could not cross between FLINT and `rug::Float`
    /// without rounding, because its exponent or mantissa is outside the range
    /// MPFR can hold.
    ///
    /// Refused rather than rounded: an inflated radius would still be a valid
    /// enclosure, but a silently *deflated* one would not, and the conversion
    /// is not in a position to tell the two apart cheaply.
    NotRepresentable {
        /// What failed to convert, e.g. `"midpoint"`.
        what: &'static str,
    },
}

impl fmt::Display for ThetaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThetaError::BackendUnavailable { capability } => write!(
                f,
                "this build's FLINT does not provide the `{capability}` layer; \
                 theta and modular functions need FLINT >= 3.1 (>= 3.2 for acb_theta)"
            ),
            ThetaError::AbiMismatch { detail } => {
                write!(f, "FLINT ball ABI mismatch: {detail}")
            }
            ThetaError::PrecisionOutOfRange { bits, max } => {
                write!(f, "working precision {bits} is outside 2..={max} bits")
            }
            ThetaError::GenusOutOfRange { genus, max } => write!(
                f,
                "genus {genus} is outside 1..={max}; theta returns 4^g values, \
                 so the output alone is exponential in the genus"
            ),
            ThetaError::DimensionMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} needs {expected} entries but {got} were given"),
            ThetaError::NotSymmetric { i, j } => write!(
                f,
                "the period matrix is not symmetric: entry ({i}, {j}) and ({j}, {i}) \
                 are different balls"
            ),
            ThetaError::NotInSiegelUpperHalfSpace { genus, prec } => write!(
                f,
                "the imaginary part of this genus-{genus} period matrix could not be \
                 proved positive definite at {prec} bits, so it is not certainly in \
                 the Siegel upper half-space"
            ),
            ThetaError::CharacteristicOutOfRange { ab, genus } => {
                write!(f, "theta characteristic {ab} is outside 0..4^{genus}")
            }
            ThetaError::NotInUpperHalfPlane { function } => write!(
                f,
                "{function} needs tau with a certainly positive imaginary part"
            ),
            ThetaError::InsufficientAccuracy {
                function,
                requested_bits,
                achieved_bits,
                at_precision,
            } => match achieved_bits {
                Some(a) => write!(
                    f,
                    "{function} reached {a} bits of relative accuracy at a working \
                     precision of {at_precision} bits, short of the {requested_bits} asked for"
                ),
                None => write!(
                    f,
                    "{function} came back with an infinite radius at a working precision \
                     of {at_precision} bits, short of the {requested_bits} bits asked for"
                ),
            },
            ThetaError::Indeterminate { function } => write!(
                f,
                "{function} could not be bounded at all: the result has an infinite radius"
            ),
            ThetaError::NotRepresentable { what } => write!(
                f,
                "the {what} has an exponent outside the range that can be moved between \
                 FLINT and MPFR without rounding"
            ),
        }
    }
}

impl std::error::Error for ThetaError {}

impl crate::errors::AlkahestError for ThetaError {
    fn code(&self) -> &'static str {
        match self {
            ThetaError::BackendUnavailable { .. } => "E-THETA-001",
            ThetaError::AbiMismatch { .. } => "E-THETA-002",
            ThetaError::PrecisionOutOfRange { .. } => "E-THETA-003",
            ThetaError::GenusOutOfRange { .. } => "E-THETA-004",
            ThetaError::DimensionMismatch { .. } => "E-THETA-005",
            ThetaError::NotSymmetric { .. } => "E-THETA-006",
            ThetaError::NotInSiegelUpperHalfSpace { .. } => "E-THETA-007",
            ThetaError::CharacteristicOutOfRange { .. } => "E-THETA-008",
            ThetaError::NotInUpperHalfPlane { .. } => "E-THETA-009",
            ThetaError::InsufficientAccuracy { .. } => "E-THETA-010",
            ThetaError::Indeterminate { .. } => "E-THETA-011",
            ThetaError::NotRepresentable { .. } => "E-THETA-012",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}
