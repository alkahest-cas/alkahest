//! Refusals for number-field construction and arithmetic.
//!
//! Every variant names the offending input. A number field is defined by an
//! *irreducible* polynomial, and the failure mode this module exists to avoid
//! is the one where a reducible polynomial is accepted, the quotient turns out
//! to be a ring with zero divisors, and an "inverse" comes back for an element
//! that has none.

use std::fmt;

/// Why a number-field construction or element operation refused.
///
/// Codes are `E-NUMF-001` … `E-NUMF-008`; see [`crate::errors::codes::REGISTRY`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NumberFieldError {
    /// The defining polynomial had no non-zero coefficients at all.
    EmptyDefiningPolynomial,
    /// The defining polynomial's degree is outside `1 ..= `[`MAX_FIELD_DEGREE`].
    ///
    /// Degree 0 is not an oversight: `ℚ[x]/(c)` for a non-zero constant `c` is
    /// the zero ring, not a field. Degree 1 *is* allowed and gives `ℚ` itself.
    ///
    /// [`MAX_FIELD_DEGREE`]: crate::numfield::MAX_FIELD_DEGREE
    DegreeOutOfRange {
        /// The degree that was offered.
        degree: i64,
        /// The largest degree this module will build.
        max: usize,
    },
    /// The defining polynomial factors over `ℚ`, so the quotient is not a field.
    ///
    /// This is the refusal the module is built around. `ℚ[x]/(f)` for a
    /// reducible `f` is a ring with zero divisors: it contains non-zero
    /// elements whose product is zero, `inverse` has no answer for them, and
    /// `norm` stops being multiplicative. Accepting it and computing anyway is
    /// the confident wrong answer this refuses to give.
    Reducible {
        /// The polynomial as offered, rendered in `x`.
        poly: String,
        /// One proper factor FLINT found, rendered in `x` — enough to show the
        /// caller *why*, without printing the whole factorisation.
        factor: String,
    },
    /// A coefficient could not be read as a rational number.
    MalformedCoefficient {
        /// The text that failed to parse.
        text: String,
    },
    /// Division by zero, or the inverse of zero.
    ///
    /// In a genuine field this is the only non-invertible element, which is
    /// exactly why the reducibility check above is not optional.
    NotInvertible,
    /// Two operands live in different number fields.
    ///
    /// Two fields are the same here when their *canonical* defining
    /// polynomials agree — primitive, integral, positive leading coefficient.
    /// `ℚ[x]/(x²−2)` and `ℚ[x]/(2x²−4)` are therefore the same field, while
    /// `ℚ[x]/(x²−2)` and `ℚ[x]/(x²−8)` are not, even though both are `ℚ(√2)`
    /// abstractly: an element's coordinates mean different things in each.
    FieldMismatch {
        /// Left-hand field, rendered.
        lhs: String,
        /// Right-hand field, rendered.
        rhs: String,
    },
    /// An element was given more coefficients than the degree of the field.
    ///
    /// An element of a degree-`d` field is a polynomial of degree `< d` in the
    /// generator. Silently reducing a longer vector would accept a typo.
    CoefficientCountMismatch {
        /// Number of coefficients supplied.
        got: usize,
        /// Degree of the field.
        degree: usize,
    },
    /// `ℚ(ζ_n)` was asked for with an `n` this module will not build.
    ///
    /// `n` must be at least 1, and `φ(n)` — the degree of the field — must be
    /// within [`MAX_FIELD_DEGREE`].
    ///
    /// [`MAX_FIELD_DEGREE`]: crate::numfield::MAX_FIELD_DEGREE
    CyclotomicOrderOutOfRange {
        /// The order that was offered.
        n: u64,
        /// The largest degree this module will build.
        max: usize,
    },
}

impl fmt::Display for NumberFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumberFieldError::EmptyDefiningPolynomial => write!(
                f,
                "the defining polynomial is zero; a number field needs a \
                 polynomial of degree at least 1"
            ),
            NumberFieldError::DegreeOutOfRange { degree, max } => write!(
                f,
                "defining polynomial of degree {degree} is outside 1..={max}; \
                 degree 0 would make Q[x]/(f) the zero ring, not a field"
            ),
            NumberFieldError::Reducible { poly, factor } => write!(
                f,
                "{poly} is reducible over Q ({factor} divides it), so Q[x]/(f) \
                 is a ring with zero divisors rather than a field"
            ),
            NumberFieldError::MalformedCoefficient { text } => {
                write!(f, "{text:?} is not a rational number")
            }
            NumberFieldError::NotInvertible => {
                write!(f, "zero has no inverse and cannot be divided by")
            }
            NumberFieldError::FieldMismatch { lhs, rhs } => write!(
                f,
                "operands live in different number fields: Q[x]/({lhs}) and Q[x]/({rhs})"
            ),
            NumberFieldError::CoefficientCountMismatch { got, degree } => write!(
                f,
                "element has {got} coefficients but the field has degree {degree}; \
                 an element is a polynomial of degree < {degree} in the generator"
            ),
            NumberFieldError::CyclotomicOrderOutOfRange { n, max } => write!(
                f,
                "Q(zeta_{n}) is out of range: n must be at least 1 and phi(n) at most {max}"
            ),
        }
    }
}

impl std::error::Error for NumberFieldError {}

impl crate::errors::AlkahestError for NumberFieldError {
    fn code(&self) -> &'static str {
        match self {
            NumberFieldError::EmptyDefiningPolynomial => "E-NUMF-001",
            NumberFieldError::DegreeOutOfRange { .. } => "E-NUMF-002",
            NumberFieldError::Reducible { .. } => "E-NUMF-003",
            NumberFieldError::MalformedCoefficient { .. } => "E-NUMF-004",
            NumberFieldError::NotInvertible => "E-NUMF-005",
            NumberFieldError::FieldMismatch { .. } => "E-NUMF-006",
            NumberFieldError::CoefficientCountMismatch { .. } => "E-NUMF-007",
            NumberFieldError::CyclotomicOrderOutOfRange { .. } => "E-NUMF-008",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}
