//! [`SpecialFunctionError`] — the refusals of [`crate::orthopoly`] and
//! [`crate::specfun`].

use std::fmt;

use crate::budget::BudgetError;
use crate::errors::AlkahestError;

/// A refusal from the orthogonal-polynomial, spherical-harmonic or
/// angular-momentum-coupling constructors.
///
/// | code | variant | meaning |
/// |---|---|---|
/// | `E-SPECFN-001` | [`NotHalfInteger`](Self::NotHalfInteger) | an angular-momentum quantum number that is neither an integer nor a half-integer (`j = 1/3`, `0.3`, a symbol) |
/// | `E-SPECFN-002` | [`NegativeAngularMomentum`](Self::NegativeAngularMomentum) | `j < 0` or `l < 0` |
/// | `E-SPECFN-003` | [`NotAnInteger`](Self::NotAnInteger) | an argument that has to be an integer is not one (`spherical_harmonic_y(1/2, …)`) |
/// | `E-SPECFN-004` | [`TooLarge`](Self::TooLarge) | an explicit expansion or an exact coupling coefficient past the size cap |
/// | `E-SPECFN-005` | [`WrongArity`](Self::WrongArity) | the wrong number of parameters for the family |
/// | `E-SPECFN-006` | [`Pole`](Self::Pole) | the function has a pole at the requested point (`zeta(1)`, `polylog(1, 1)`) |
/// | `E-SPECFN-007` | [`OrderNotInteger`](Self::OrderNotInteger) | an order that must be an integer literal is not one (`spherical_jn(1/2, x)`) |
/// | `E-BUDGET-*` | [`Budget`](Self::Budget) | the active `Budget` ran out |
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpecialFunctionError {
    /// An angular-momentum argument is not an integer or a half-integer.
    NotHalfInteger { what: String, value: String },
    /// An angular momentum `j` (or orbital `l`) is negative.
    NegativeAngularMomentum { what: String, value: String },
    /// An argument that must be an integer literal is not one.
    NotAnInteger { what: String, value: String },
    /// The requested object is past the size this module will build exactly.
    TooLarge { what: String, value: i64, max: i64 },
    /// A family was given the wrong number of parameters.
    WrongArity {
        family: &'static str,
        expected: usize,
        got: usize,
    },
    /// The function has a pole at the requested argument (`ζ(1)`).
    Pole { function: &'static str, at: String },
    /// An order that has to be an integer literal is not one.
    OrderNotInteger {
        function: &'static str,
        value: String,
    },
    /// The active [`crate::budget::Budget`] ran out mid-computation.
    Budget(BudgetError),
}

impl fmt::Display for SpecialFunctionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecialFunctionError::NotHalfInteger { what, value } => write!(
                f,
                "{what} = {value} is not an integer or a half-integer; angular-momentum \
                 quantum numbers take values in ½ℤ"
            ),
            SpecialFunctionError::NegativeAngularMomentum { what, value } => {
                write!(f, "{what} = {value} is negative; an angular momentum is ≥ 0")
            }
            SpecialFunctionError::NotAnInteger { what, value } => {
                write!(f, "{what} = {value} must be an integer literal")
            }
            SpecialFunctionError::TooLarge { what, value, max } => write!(
                f,
                "{what} = {value} is past the exact-construction cap of {max}"
            ),
            SpecialFunctionError::WrongArity {
                family,
                expected,
                got,
            } => write!(
                f,
                "{family} takes {expected} parameter(s) between the degree and the variable, got {got}"
            ),
            SpecialFunctionError::Pole { function, at } => {
                write!(f, "{function} has a pole at {at}; it has no finite value there")
            }
            SpecialFunctionError::OrderNotInteger { function, value } => {
                write!(f, "{function}: the order {value} must be an integer")
            }
            SpecialFunctionError::Budget(b) => write!(f, "{b}"),
        }
    }
}

impl std::error::Error for SpecialFunctionError {}

impl AlkahestError for SpecialFunctionError {
    fn code(&self) -> &'static str {
        match self {
            SpecialFunctionError::NotHalfInteger { .. } => "E-SPECFN-001",
            SpecialFunctionError::NegativeAngularMomentum { .. } => "E-SPECFN-002",
            SpecialFunctionError::NotAnInteger { .. } => "E-SPECFN-003",
            SpecialFunctionError::TooLarge { .. } => "E-SPECFN-004",
            SpecialFunctionError::WrongArity { .. } => "E-SPECFN-005",
            SpecialFunctionError::Pole { .. } => "E-SPECFN-006",
            SpecialFunctionError::OrderNotInteger { .. } => "E-SPECFN-007",
            SpecialFunctionError::Budget(b) => b.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            SpecialFunctionError::NotHalfInteger { .. } => Some(
                "pass integers or half-integers (ints, Fractions such as Fraction(1, 2), \
                 or floats that are exactly k/2)",
            ),
            SpecialFunctionError::NegativeAngularMomentum { .. } => Some(
                "angular momenta j and l are non-negative; only the projections m may be negative",
            ),
            SpecialFunctionError::NotAnInteger { .. } => Some(
                "spherical harmonics are built explicitly for integer l and m only; substitute \
                 integers before calling",
            ),
            SpecialFunctionError::TooLarge { .. } => Some(
                "keep the polynomial symbolic (expand=False) and evaluate it numerically, \
                 or ask for a smaller degree / angular momentum",
            ),
            SpecialFunctionError::WrongArity { .. } => {
                Some("check the family's signature, e.g. jacobi_p(n, a, b, x)")
            }
            SpecialFunctionError::Pole { .. } => Some(
                "evaluate at a different point, or take the limit / Laurent expansion \
                 there (zeta(s) ~ 1/(s - 1) as s -> 1)",
            ),
            SpecialFunctionError::OrderNotInteger { .. } => Some(
                "spherical Bessel functions are built for integer order n; use \
                 bessel_j(n + 1/2, x) for a general order",
            ),
            SpecialFunctionError::Budget(b) => b.remediation(),
        }
    }
}

impl From<BudgetError> for SpecialFunctionError {
    fn from(b: BudgetError) -> Self {
        SpecialFunctionError::Budget(b)
    }
}
