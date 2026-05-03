use std::fmt;

/// Reason an expression could not be converted to a polynomial representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    /// A symbol other than the target variable(s) appeared as a free term.
    UnexpectedSymbol(String),
    /// A rational or float coefficient was encountered; only integers are allowed.
    NonIntegerCoefficient,
    /// An exponent was a negative integer (produces a rational function, not a poly).
    NegativeExponent,
    /// An exponent was a non-negative integer too large to fit in u32.
    ExponentTooLarge,
    /// An exponent was not a constant integer (e.g. symbolic or float).
    NonConstantExponent,
    /// A function call appeared and may contain the variable.
    NonPolynomialFunction(String),
    /// The denominator of a RationalFunction would be zero.
    ZeroDenominator,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::UnexpectedSymbol(s) => {
                write!(f, "unexpected free symbol '{s}' in polynomial expression")
            }
            ConversionError::NonIntegerCoefficient => {
                write!(
                    f,
                    "non-integer coefficient (rational or float) in polynomial"
                )
            }
            ConversionError::NegativeExponent => {
                write!(
                    f,
                    "negative exponent yields a rational function, not a polynomial"
                )
            }
            ConversionError::ExponentTooLarge => {
                write!(f, "exponent exceeds u32::MAX")
            }
            ConversionError::NonConstantExponent => {
                write!(f, "exponent is not a constant integer")
            }
            ConversionError::NonPolynomialFunction(name) => {
                write!(f, "function '{name}' cannot appear in a polynomial")
            }
            ConversionError::ZeroDenominator => {
                write!(f, "denominator is zero")
            }
        }
    }
}

impl std::error::Error for ConversionError {}

impl crate::errors::AlkahestError for ConversionError {
    fn code(&self) -> &'static str {
        match self {
            ConversionError::UnexpectedSymbol(_) => "E-POLY-001",
            ConversionError::NonIntegerCoefficient => "E-POLY-002",
            ConversionError::NegativeExponent => "E-POLY-003",
            ConversionError::ExponentTooLarge => "E-POLY-004",
            ConversionError::NonConstantExponent => "E-POLY-005",
            ConversionError::NonPolynomialFunction(_) => "E-POLY-006",
            ConversionError::ZeroDenominator => "E-POLY-007",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        ConversionError::remediation(self)
    }
}

impl ConversionError {
    /// A human-readable remediation hint for the user.
    pub fn remediation(&self) -> Option<&'static str> {
        match self {
            ConversionError::NonPolynomialFunction(_) => Some(
                "not a polynomial; wrap in the function only after rational reduction",
            ),
            ConversionError::NegativeExponent => Some(
                "negative exponent yields a rational function; use RationalFunction::from_symbolic instead",
            ),
            ConversionError::NonConstantExponent => Some(
                "symbolic exponents are not supported; substitute concrete values first",
            ),
            ConversionError::NonIntegerCoefficient => Some(
                "rational/float coefficients not allowed; multiply through by the denominator",
            ),
            ConversionError::UnexpectedSymbol(_) => Some(
                "pass all free variables in the `vars` argument to poly_normal",
            ),
            ConversionError::ExponentTooLarge => Some(
                "exponent exceeds u32::MAX; consider working with rational functions",
            ),
            ConversionError::ZeroDenominator => None,
        }
    }

    /// Optional source span `(start_byte, end_byte)` within the input text.
    /// `None` until the parser is integrated.
    pub fn span(&self) -> Option<(usize, usize)> {
        None
    }
}

/// Failure modes for polynomial factorization (V2-7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorError {
    /// The zero polynomial has no multiplicative factorization.
    ZeroPolynomial,
    /// Modulus must be an integer ≥ 2 for 𝔽_p factoring.
    InvalidModulus,
    /// FLINT returned an error (rare for well-formed input).
    FlintFailure,
}

impl fmt::Display for FactorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactorError::ZeroPolynomial => write!(f, "cannot factor the zero polynomial"),
            FactorError::InvalidModulus => write!(f, "modulus must be at least 2"),
            FactorError::FlintFailure => {
                write!(f, "polynomial factorization failed internally (FLINT)")
            }
        }
    }
}

impl std::error::Error for FactorError {}

impl crate::errors::AlkahestError for FactorError {
    fn code(&self) -> &'static str {
        match self {
            FactorError::ZeroPolynomial => "E-POLY-008",
            FactorError::InvalidModulus => "E-POLY-009",
            FactorError::FlintFailure => "E-POLY-010",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            FactorError::ZeroPolynomial => "factorization is only defined for non-zero polynomials",
            FactorError::InvalidModulus => {
                "use a modulus ≥ 2 that fits in a machine word (FLINT `nmod`)"
            }
            FactorError::FlintFailure => {
                "report the polynomial as a minimal failing example"
            }
        })
    }
}
