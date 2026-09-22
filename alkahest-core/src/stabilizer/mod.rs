//! The **binary symplectic / stabilizer-code** layer: Pauli groups, stabilizer
//! codes, CSS codes, and the classical matrix groups `GL`, `SL` and `Sp` over
//! GF(q).
//!
//! This module is built on [`crate::ffield`] (dense GF(q) linear algebra) and
//! [`crate::group`] (permutation groups). Nothing here reimplements Gaussian
//! elimination or Schreier–Sims; every rank, nullspace and group order is
//! delegated.
//!
//! # The convention, stated once and loudly
//!
//! A Pauli operator on `n` qubits is written in the **`(x | z)` layout**: a
//! single vector of `2n` bits whose first `n` entries are the `X` exponents and
//! whose last `n` entries are the `Z` exponents, in that order. The symplectic
//! form is
//!
//! ```text
//!     ⟨(x₁ | z₁), (x₂ | z₂)⟩  =  x₁·z₂  +  z₁·x₂         (mod 2)
//! ```
//!
//! and two Paulis **commute exactly when this form vanishes**. As a matrix,
//! `⟨u, v⟩ = u Ω vᵀ` with
//!
//! ```text
//!     Ω = [ 0  I ]        (block form, each block n × n)
//!         [ I  0 ]
//! ```
//!
//! which over GF(2) is the same matrix as the usual `[[0, I], [−I, 0]]`, since
//! `−1 = 1`. This is the single most common source of wrong answers in this
//! area, so three consequences are spelled out rather than left implicit:
//!
//! * **`(z | x)` is a different convention**, used by some of the literature and
//!   by some libraries. A check matrix transcribed from a paper that uses it
//!   will produce a group that commutes, has the right rank, and encodes the
//!   wrong code. There is no way for this module to detect that, so there is no
//!   automatic repair: the layout is part of the input's meaning.
//! * **The letter `Y` in a [`PauliOperator`]'s `Display` means the symplectic
//!   pair `(x, z) = (1, 1)`, i.e. the operator `XZ`, and `XZ = −i·Y`.** The
//!   remaining phase lives in the operator's `Z₄` phase field and is printed as
//!   a prefix. So `-i·XY` is a genuine Pauli and not a typo.
//! * **`Sp(2n, 2)` membership is tested as `MᵀΩM = Ω`**, the column-vector
//!   convention. Over GF(2) — where `Ω² = I` — that is equivalent to the
//!   row-vector condition `MΩMᵀ = Ω`, and
//!   [`symplectic::is_symplectic`] asserts the equivalence in its own tests
//!   rather than asking the reader to take it on faith.
//!
//! # What is here
//!
//! * [`symplectic`] — the form itself, [`symplectic::symplectic_gram_schmidt`]
//!   (a hyperbolic basis plus the radical of the restricted form),
//!   [`symplectic::symplectic_complement`], and
//!   [`symplectic::is_symplectic`].
//! * [`PauliOperator`] — `(x | z)` bits and a phase in `Z₄`, with
//!   multiplication, [`PauliOperator::commutes_with`], weight and `Display`.
//! * [`StabilizerGroup`] — generators that are **checked** to commute pairwise
//!   (a refusal, [`StabilizerError::NotCommuting`], never a silent fixup), plus
//!   rank, independence, and the check that `−I` is not in the group.
//! * [`StabilizerCode`] — `n`, `k = n − rank(S)`, the `k` logical `X`/`Z` pairs
//!   from symplectic Gram–Schmidt on the centralizer, and the syndrome map.
//! * [`CssCode`] — from `(H_X, H_Z)` over GF(2), refusing unless
//!   `H_X · H_Zᵀ = 0`.
//! * [`MatrixGroup`] — `GL(n, q)`, `SL(n, q)`, `Sp(2n, q)`: exact order from the
//!   standard product formulas as a [`rug::Integer`], membership testing, and
//!   (below a cap) the induced permutation action on non-zero vectors, handed
//!   to [`crate::group::PermutationGroup`].
//!
//! # What is deliberately *not* here
//!
//! * **Classical linear codes.** There is no `LinearCode` type in this module
//!   on purpose; classical codes, weight enumerators and LP bounds belong to
//!   the `coding` layer. Everything here takes `GfMatrix` check matrices
//!   directly.
//! * **Qudit (non-binary) stabilizer codes.** The Pauli and stabilizer surface
//!   is GF(2) only, and a non-binary field is refused with
//!   [`StabilizerError::FieldNotBinary`]. Only [`MatrixGroup`] is defined over
//!   general GF(q).
//! * **Clifford-circuit simulation and tableau updates.** This is the algebra
//!   of the stabilizer formalism, not a simulator: there is no
//!   `apply_hadamard`, no `apply_cnot`, no measurement.
//! * **Decoding.** [`StabilizerCode::syndrome`] maps an error to its syndrome.
//!   The inverse problem — minimum-weight perfect matching, BP+OSD, union-find
//!   — is not implemented, and no function here pretends to solve it.
//! * **Distance for codes of interesting size.** Minimum distance is
//!   `NP`-hard, and the only algorithm here is exhaustive search over the
//!   centralizer, which costs `2^(n+k)`. Above
//!   [`MAX_DISTANCE_SEARCH_DIM`] it is a typed refusal
//!   ([`StabilizerError::DistanceSearchTooLarge`]), never an estimate. When
//!   only a bound is available, it comes back as
//!   [`Distance::UpperBound`] — a different variant, so that a caller cannot
//!   read a bound as a distance by accident.
//! * **Subsystem / gauge codes, and non-stabilizer codes.** Only the
//!   `[[n, k, d]]` stabilizer class.
//!
//! # Worked example — the Steane `[[7, 1, 3]]` code
//!
//! ```
//! use alkahest_cas::experimental::{CssCode, Distance, FiniteField, GfMatrix};
//!
//! let gf2 = FiniteField::prime(2).unwrap();
//! // The [7,4,3] Hamming parity-check matrix, used as both H_X and H_Z.
//! let h = GfMatrix::from_u64(&gf2, 3, 7, &[
//!     0, 0, 0, 1, 1, 1, 1,
//!     0, 1, 1, 0, 0, 1, 1,
//!     1, 0, 1, 0, 1, 0, 1,
//! ]).unwrap();
//!
//! let code = CssCode::new(&h, &h).unwrap();
//! assert_eq!(code.n(), 7);
//! assert_eq!(code.k(), 1);
//! assert_eq!(code.minimum_distance().unwrap(), Distance::Exact(3));
//! ```

pub mod code;
pub mod css;
pub mod matrix_group;
pub mod pauli;
pub mod symplectic;

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

pub use code::{Distance, StabilizerCode, MAX_DISTANCE_SEARCH_DIM};
pub use css::CssCode;
pub use matrix_group::{MatrixGroup, MatrixGroupKind, MAX_MATRIX_ENUMERATION};
pub use pauli::{PauliOperator, StabilizerGroup};
pub use symplectic::{
    is_symplectic, symplectic_complement, symplectic_form, symplectic_gram_matrix,
    symplectic_gram_schmidt, HyperbolicBasis,
};

use crate::ffield::FiniteFieldError;
use std::fmt;

/// The largest qubit count this module will build a Pauli or a code on.
///
/// The bound is not a mathematical one. Every symplectic vector here is a
/// `Vec<u8>` of length `2n` and every linear-algebra step goes through a dense
/// `2n`-column [`crate::ffield::GfMatrix`], so the centralizer computation is
/// `O(n³)` words. A cap keeps a typo (`n = 10_000_000`) from becoming an
/// allocation failure with no diagnosis.
pub const MAX_QUBITS: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a symplectic, Pauli, stabilizer-code or matrix-group operation refused.
///
/// Every variant is a refusal. This module has no "best effort" path: a
/// distance that could not be verified is not returned as a number, and a set
/// of generators that does not commute is not silently repaired by dropping
/// one.
///
/// [`StabilizerError::FiniteField`] is the one variant that is not a refusal of
/// this module's own: it carries a [`FiniteFieldError`] out of the GF(q) layer
/// unchanged, **including its `E-GFQ-NNN` code**, so that a caller branching on
/// `.code()` sees where the failure really happened rather than a re-labelled
/// `E-STAB-*`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StabilizerError {
    /// The Pauli / stabilizer surface was handed a field other than GF(2).
    ///
    /// Qudit stabilizer codes over GF(q) are a real and different theory (the
    /// form is `Z_q`-valued, the Pauli group's phases live in `Z_{2q}` or
    /// `Z_q`, and "weight" and "distance" are defined against a different
    /// alphabet). Reinterpreting a GF(q) matrix as if its entries were bits
    /// would produce a well-typed wrong answer, so it is refused.
    FieldNotBinary {
        /// The field that was offered, rendered.
        field: String,
    },
    /// A vector or matrix has the wrong length or shape for the operation.
    ShapeMismatch {
        /// The operation that was attempted, e.g. `"symplectic form"`.
        op: &'static str,
        /// What the operation needed, in words: `"length 2n = 14"`.
        expected: String,
        /// What it got.
        got: String,
    },
    /// Two Pauli operators are stated on different numbers of qubits.
    ///
    /// Qubit counts are **not** widened by padding with identities, for the
    /// same reason permutation degrees are not padded with fixed points in
    /// [`crate::group`]: which embedding the caller meant is not recoverable
    /// from the arguments.
    QubitCountMismatch {
        /// Qubit count of the left operand.
        left: usize,
        /// Qubit count of the right operand.
        right: usize,
    },
    /// Two proposed stabilizer generators anticommute.
    ///
    /// A stabilizer group is an **abelian** subgroup of the Pauli group. A
    /// pair that anticommutes has no common `+1` eigenspace, so the "code"
    /// these generators describe is the zero space. Dropping or negating one
    /// generator would produce *a* code, just not the one that was asked for.
    NotCommuting {
        /// Index of the first generator, 0-based, in the list as supplied.
        i: usize,
        /// Index of the second.
        j: usize,
    },
    /// `H_X · H_Zᵀ ≠ 0`: the pair of check matrices is not a CSS pair.
    ///
    /// The `X`-type stabilizer from row `row` of `H_X` and the `Z`-type
    /// stabilizer from row `col` of `H_Z` anticommute, so they do not generate
    /// an abelian group. This is the CSS condition, and it is checked rather
    /// than assumed.
    CssConditionViolated {
        /// Row of `H_X` whose stabilizer anticommutes.
        row: usize,
        /// Row of `H_Z` whose stabilizer it anticommutes with.
        col: usize,
    },
    /// A product of the supplied generators is `−I`.
    ///
    /// Then the `+1` eigenspace is `{0}` and the code encodes nothing. This is
    /// reachable only from *dependent* generators: if the symplectic vectors
    /// are linearly independent, no non-trivial product can have trivial
    /// symplectic part at all.
    ContainsMinusIdentity {
        /// The indices of the generators whose product is `−I`.
        combination: Vec<usize>,
    },
    /// The code has `k = 0`, so "minimum distance" names nothing.
    ///
    /// Distance is the least weight of an element of `N(S) \ S` — an
    /// undetectable error. With no logical qubits that set is empty, and the
    /// conventional answers (`n`, `∞`, `0`) are three different conventions,
    /// none of them a fact about the code.
    NoLogicalQubits {
        /// The code's block length, for the message.
        n: usize,
    },
    /// The exhaustive distance search would visit more than the cap allows.
    ///
    /// Minimum distance of a stabilizer code is `NP`-hard; the only algorithm
    /// here enumerates the `2^(n+k)` elements of the centralizer. Above the cap
    /// there is no answer, only an [`Distance::UpperBound`] from
    /// [`StabilizerCode::distance_upper_bound`].
    DistanceSearchTooLarge {
        /// `n + k`, the dimension of the space that would be enumerated.
        dim: usize,
        /// The cap in force.
        cap: usize,
    },
    /// The qubit count exceeds [`MAX_QUBITS`].
    TooManyQubits {
        /// The count that was asked for.
        n: usize,
        /// The largest this module accepts.
        max: usize,
    },
    /// A Pauli operator could not be built from the values given.
    MalformedPauli {
        /// What is wrong: a bit outside `{0, 1}`, or a phase outside `Z₄`.
        reason: String,
    },
    /// A matrix group is too large to list element by element.
    ///
    /// The order is still exact and still available from
    /// [`MatrixGroup::order`]: it is the *list* that is refused. Enumeration
    /// here is by brute force over all `q^(d²)` matrices of the right size,
    /// which is what the cap counts.
    EnumerationTooLarge {
        /// Decimal rendering of the number of candidate matrices.
        candidates: String,
        /// The cap that was in force.
        cap: u64,
    },
    /// Two operands live over different finite fields.
    FieldMismatch {
        /// Left-hand field, rendered.
        lhs: String,
        /// Right-hand field, rendered.
        rhs: String,
    },
    /// A result was computed, failed its own consistency check, and was
    /// **withheld** rather than returned.
    ///
    /// Reaching this is a bug in this module, not in the input. It exists so
    /// that the bug surfaces as an error with a code instead of as a set of
    /// "logical operators" that do not commute with the stabilizer.
    InconsistentResult {
        /// Which invariant failed, concretely.
        reason: String,
    },
    /// The GF(q) layer refused. The inner error's `E-GFQ-NNN` code is
    /// preserved by [`crate::errors::AlkahestError::code`].
    FiniteField(FiniteFieldError),
}

impl From<FiniteFieldError> for StabilizerError {
    fn from(e: FiniteFieldError) -> Self {
        StabilizerError::FiniteField(e)
    }
}

impl fmt::Display for StabilizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StabilizerError::FieldNotBinary { field } => write!(
                f,
                "the Pauli and stabilizer surface is defined over GF(2) only; got {field}. \
                 Qudit stabilizer codes are a different theory, not a generalisation applied here"
            ),
            StabilizerError::ShapeMismatch { op, expected, got } => {
                write!(f, "{op} needs {expected}, but got {got}")
            }
            StabilizerError::QubitCountMismatch { left, right } => write!(
                f,
                "Pauli operators are on {left} and {right} qubits; counts are not padded \
                 with identities automatically"
            ),
            StabilizerError::NotCommuting { i, j } => write!(
                f,
                "generators {i} and {j} anticommute, so they generate no abelian group \
                 and stabilize no non-zero state"
            ),
            StabilizerError::CssConditionViolated { row, col } => write!(
                f,
                "the CSS condition H_X · H_Zᵀ = 0 fails at entry ({row}, {col}): the X-type \
                 stabilizer from row {row} of H_X anticommutes with the Z-type stabilizer \
                 from row {col} of H_Z"
            ),
            StabilizerError::ContainsMinusIdentity { combination } => write!(
                f,
                "the product of generators {combination:?} is −I, so the +1 eigenspace \
                 is {{0}} and the code encodes nothing"
            ),
            StabilizerError::NoLogicalQubits { n } => write!(
                f,
                "this [[{n}, 0]] code has no logical qubits, so it has no minimum distance: \
                 N(S) \\ S is empty"
            ),
            StabilizerError::DistanceSearchTooLarge { dim, cap } => write!(
                f,
                "an exhaustive distance search would enumerate 2^{dim} centralizer elements, \
                 past the cap of 2^{cap}"
            ),
            StabilizerError::TooManyQubits { n, max } => {
                write!(f, "{n} qubits exceeds this module's limit of {max}")
            }
            StabilizerError::MalformedPauli { reason } => {
                write!(f, "malformed Pauli operator: {reason}")
            }
            StabilizerError::EnumerationTooLarge { candidates, cap } => write!(
                f,
                "listing this group would filter {candidates} candidate matrices, past the \
                 cap of {cap}"
            ),
            StabilizerError::FieldMismatch { lhs, rhs } => {
                write!(f, "operands live over different fields: {lhs} and {rhs}")
            }
            StabilizerError::InconsistentResult { reason } => write!(
                f,
                "a result failed its own consistency check and was withheld: {reason}. \
                 This is a bug in alkahest's stabilizer layer, not in the input"
            ),
            StabilizerError::FiniteField(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StabilizerError {}

impl crate::errors::AlkahestError for StabilizerError {
    fn code(&self) -> &'static str {
        match self {
            StabilizerError::FieldNotBinary { .. } => "E-STAB-001",
            StabilizerError::ShapeMismatch { .. } => "E-STAB-002",
            StabilizerError::QubitCountMismatch { .. } => "E-STAB-003",
            StabilizerError::NotCommuting { .. } => "E-STAB-004",
            StabilizerError::CssConditionViolated { .. } => "E-STAB-005",
            StabilizerError::ContainsMinusIdentity { .. } => "E-STAB-006",
            StabilizerError::NoLogicalQubits { .. } => "E-STAB-007",
            StabilizerError::DistanceSearchTooLarge { .. } => "E-STAB-008",
            StabilizerError::TooManyQubits { .. } => "E-STAB-009",
            StabilizerError::MalformedPauli { .. } => "E-STAB-010",
            StabilizerError::EnumerationTooLarge { .. } => "E-STAB-011",
            StabilizerError::FieldMismatch { .. } => "E-STAB-012",
            StabilizerError::InconsistentResult { .. } => "E-STAB-013",
            // Delegated on purpose: the GF(q) layer's own code survives the
            // boundary, so `.code()` still says where the failure happened.
            StabilizerError::FiniteField(e) => crate::errors::AlkahestError::code(e),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

use crate::ffield::{FiniteField, GfMatrix};
use std::sync::OnceLock;

/// GF(2), built once.
pub(crate) fn gf2() -> &'static FiniteField {
    static GF2: OnceLock<FiniteField> = OnceLock::new();
    GF2.get_or_init(|| FiniteField::prime(2).expect("2 is prime"))
}

/// Refuse a matrix that is not over GF(2).
pub(crate) fn require_gf2(m: &GfMatrix) -> Result<(), StabilizerError> {
    if m.field().characteristic() == 2 && m.field().degree() == 1 {
        Ok(())
    } else {
        Err(StabilizerError::FieldNotBinary {
            field: format!("{}", m.field()),
        })
    }
}

/// Refuse a qubit count past [`MAX_QUBITS`].
pub(crate) fn check_qubits(n: usize) -> Result<(), StabilizerError> {
    if n > MAX_QUBITS {
        return Err(StabilizerError::TooManyQubits { n, max: MAX_QUBITS });
    }
    Ok(())
}

/// Read a GF(2) matrix out as rows of `0`/`1` bytes.
pub(crate) fn matrix_rows(m: &GfMatrix) -> Result<Vec<Vec<u8>>, StabilizerError> {
    require_gf2(m)?;
    let (r, c) = m.shape();
    let flat = m.to_u64().ok_or_else(|| StabilizerError::FieldNotBinary {
        field: format!("{}", m.field()),
    })?;
    Ok((0..r)
        .map(|i| (0..c).map(|j| (flat[i * c + j] & 1) as u8).collect())
        .collect())
}

/// Read a GF(2) matrix out as *columns* of `0`/`1` bytes.
///
/// [`GfMatrix::nullspace`] returns a basis in the columns, which is why this
/// exists beside [`matrix_rows`].
pub(crate) fn matrix_cols(m: &GfMatrix) -> Result<Vec<Vec<u8>>, StabilizerError> {
    require_gf2(m)?;
    let (r, c) = m.shape();
    let flat = m.to_u64().ok_or_else(|| StabilizerError::FieldNotBinary {
        field: format!("{}", m.field()),
    })?;
    Ok((0..c)
        .map(|j| (0..r).map(|i| (flat[i * c + j] & 1) as u8).collect())
        .collect())
}

/// Build a GF(2) matrix whose rows are the given bit vectors.
pub(crate) fn matrix_from_rows(
    rows: &[Vec<u8>],
    ncols: usize,
) -> Result<GfMatrix, StabilizerError> {
    let flat: Vec<u64> = rows
        .iter()
        .flat_map(|r| r.iter().map(|&b| u64::from(b & 1)))
        .collect();
    for r in rows {
        if r.len() != ncols {
            return Err(StabilizerError::ShapeMismatch {
                op: "assemble a matrix from rows",
                expected: format!("every row of length {ncols}"),
                got: format!("a row of length {}", r.len()),
            });
        }
    }
    Ok(GfMatrix::from_u64(gf2(), rows.len(), ncols, &flat)?)
}

/// The GF(2) rank of a set of bit vectors, by row reduction.
pub(crate) fn bit_rank(rows: &[Vec<u8>], ncols: usize) -> Result<usize, StabilizerError> {
    if rows.is_empty() || ncols == 0 {
        return Ok(0);
    }
    Ok(matrix_from_rows(rows, ncols)?.rank())
}

/// `u + v` over GF(2), componentwise.
pub(crate) fn bit_add(u: &[u8], v: &[u8]) -> Vec<u8> {
    u.iter().zip(v).map(|(a, b)| (a ^ b) & 1).collect()
}
