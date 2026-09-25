//! Refusals for finite-field construction and linear algebra.
//!
//! Every variant carries enough context to name the offending input, because
//! the alternative — a `bool` or a `None` — is exactly the shape that turns a
//! shape mismatch into a confident wrong answer three calls later.

use std::fmt;

/// Why a GF(q) construction or matrix operation refused.
///
/// Codes are `E-GFQ-001` … `E-GFQ-012`; see [`crate::errors::codes::REGISTRY`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FiniteFieldError {
    /// The characteristic is not a prime.
    ///
    /// GF(q) exists only for `q = p^k` with `p` prime. ℤ/nℤ for composite `n`
    /// is a ring with zero divisors: Gaussian elimination in it can hit a
    /// pivot that is non-zero and still not invertible, and every "rank" it
    /// reports is a guess. Refusing is the only honest answer.
    NotPrime {
        /// The modulus that was offered, rendered in decimal.
        ///
        /// A string rather than an integer because this error is reachable
        /// from a front end with arbitrary-precision integers, and naming the
        /// value the caller actually passed is worth more than being able to
        /// do arithmetic on it in an error handler.
        modulus: String,
    },
    /// The characteristic does not fit in a machine word.
    ///
    /// The FLINT backend (`nmod`/`fq_nmod`) is word-sized by construction.
    /// Multi-precision prime fields would need `fmpz_mod_mat`, which this
    /// module does not wrap.
    ModulusTooLarge {
        /// The modulus that was offered, rendered in decimal.
        modulus: String,
    },
    /// The extension degree `k` is outside `1 ..= `[`MAX_EXTENSION_DEGREE`].
    ///
    /// [`MAX_EXTENSION_DEGREE`]: crate::ffield::MAX_EXTENSION_DEGREE
    DegreeOutOfRange {
        /// The degree that was offered.
        degree: i64,
    },
    /// A caller-supplied defining polynomial cannot define GF(p^k).
    BadDefiningPolynomial {
        /// What is wrong with it: degree, reducibility, or a zero leading
        /// coefficient.
        reason: String,
    },
    /// An element was given more coefficients than the extension degree.
    ///
    /// An element of GF(p^k) is a polynomial of degree `< k` over GF(p).
    /// Silently reducing a longer vector modulo the defining polynomial would
    /// accept `[1, 0, 0, 1]` in GF(2³) as if the caller meant `a³ + 1 = a + 1`,
    /// which is almost never what a typo meant.
    MalformedElement {
        /// Number of coefficients supplied.
        got: usize,
        /// Extension degree of the field.
        degree: usize,
    },
    /// Two operands live over different fields.
    FieldMismatch {
        /// Left-hand field, rendered.
        lhs: String,
        /// Right-hand field, rendered.
        rhs: String,
    },
    /// Operand shapes do not conform.
    DimensionMismatch {
        /// The operation that was attempted, e.g. `"multiply"`.
        op: &'static str,
        /// Left-hand shape as `(rows, cols)`.
        lhs: (usize, usize),
        /// Right-hand shape as `(rows, cols)`.
        rhs: (usize, usize),
    },
    /// The operation is defined only for square matrices.
    NotSquare {
        /// The operation that was attempted, e.g. `"determinant"`.
        op: &'static str,
        /// Shape as `(rows, cols)`.
        shape: (usize, usize),
    },
    /// The matrix is singular over GF(q), so it has no inverse.
    Singular {
        /// Shape as `(rows, cols)`.
        shape: (usize, usize),
    },
    /// The linear system has no solution at all — `b` is outside the column
    /// space of `A`.
    Inconsistent,
    /// An index is outside the matrix.
    IndexOutOfBounds {
        /// Row index requested.
        i: usize,
        /// Column index requested.
        j: usize,
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// The requested shape exceeds what the FLINT backend can address.
    DimensionTooLarge {
        /// Row count requested.
        rows: usize,
        /// Column count requested.
        cols: usize,
    },
}

impl fmt::Display for FiniteFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FiniteFieldError::NotPrime { modulus } => write!(
                f,
                "{modulus} is not prime: GF(q) requires a prime characteristic"
            ),
            FiniteFieldError::ModulusTooLarge { modulus } => write!(
                f,
                "characteristic {modulus} does not fit in a 64-bit machine word"
            ),
            FiniteFieldError::DegreeOutOfRange { degree } => write!(
                f,
                "extension degree {degree} is outside 1..={}",
                super::MAX_EXTENSION_DEGREE
            ),
            FiniteFieldError::BadDefiningPolynomial { reason } => {
                write!(f, "unusable defining polynomial: {reason}")
            }
            FiniteFieldError::MalformedElement { got, degree } => write!(
                f,
                "element has {got} coefficients but the field has degree {degree}; \
                 an element of GF(p^k) is a polynomial of degree < k"
            ),
            FiniteFieldError::FieldMismatch { lhs, rhs } => {
                write!(f, "operands live over different fields: {lhs} and {rhs}")
            }
            FiniteFieldError::DimensionMismatch { op, lhs, rhs } => write!(
                f,
                "cannot {op} a {}x{} matrix with a {}x{} matrix",
                lhs.0, lhs.1, rhs.0, rhs.1
            ),
            FiniteFieldError::NotSquare { op, shape } => write!(
                f,
                "{op} requires a square matrix; this one is {}x{}",
                shape.0, shape.1
            ),
            FiniteFieldError::Singular { shape } => write!(
                f,
                "the {}x{} matrix is singular over this field and has no inverse",
                shape.0, shape.1
            ),
            FiniteFieldError::Inconsistent => write!(
                f,
                "the linear system has no solution: the right-hand side is outside \
                 the column space"
            ),
            FiniteFieldError::IndexOutOfBounds { i, j, rows, cols } => {
                write!(f, "index ({i}, {j}) is outside a {rows}x{cols} matrix")
            }
            FiniteFieldError::DimensionTooLarge { rows, cols } => write!(
                f,
                "a {rows}x{cols} matrix exceeds what the FLINT backend can address"
            ),
        }
    }
}

impl std::error::Error for FiniteFieldError {}

impl crate::errors::AlkahestError for FiniteFieldError {
    fn code(&self) -> &'static str {
        match self {
            FiniteFieldError::NotPrime { .. } => "E-GFQ-001",
            FiniteFieldError::ModulusTooLarge { .. } => "E-GFQ-002",
            FiniteFieldError::DegreeOutOfRange { .. } => "E-GFQ-003",
            FiniteFieldError::BadDefiningPolynomial { .. } => "E-GFQ-004",
            FiniteFieldError::MalformedElement { .. } => "E-GFQ-005",
            FiniteFieldError::FieldMismatch { .. } => "E-GFQ-006",
            FiniteFieldError::DimensionMismatch { .. } => "E-GFQ-007",
            FiniteFieldError::NotSquare { .. } => "E-GFQ-008",
            FiniteFieldError::Singular { .. } => "E-GFQ-009",
            FiniteFieldError::Inconsistent => "E-GFQ-010",
            FiniteFieldError::IndexOutOfBounds { .. } => "E-GFQ-011",
            FiniteFieldError::DimensionTooLarge { .. } => "E-GFQ-012",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}
