//! Typed refusals for the function-field API.
//!
//! Every variant carries a stable `E-FFLD-NNN` code registered in
//! [`crate::errors::codes::REGISTRY`].  The split between a **refusal** (the
//! implementation cannot answer) and a **verdict** (the answer is "no") is
//! deliberate and is documented per variant: [`FunctionFieldError::NonTorsion`]
//! is a verdict, everything else is a refusal.

use std::fmt;

/// Why a function-field operation refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionFieldError {
    /// The defining polynomial is not a curve model this module implements.
    ///
    /// Only `c₂·y² + c₁(x)·y + c₀(x)` with `c₂` a non-zero **rational constant**
    /// is accepted; it is normalised to `y² = a(x)` with `a` squarefree of
    /// degree ≥ 1.  Anything else — `deg_y f ≠ 2`, a non-constant leading
    /// coefficient, a discriminant that is zero or a non-zero constant — is
    /// outside the modelled class.
    UnsupportedModel {
        /// What specifically failed.
        reason: String,
    },
    /// The curve is a **real** (even-degree) hyperelliptic model, and this
    /// operation is implemented only for the **imaginary** (odd-degree) one.
    ///
    /// `y² = a(x)` with `deg a = 2g+2` has **two** places above `x = ∞`.  The
    /// Mumford/Cantor representation reused here (`integrate::algebraic`) fixes
    /// a single rational place at infinity as the base point of every class, so
    /// it does not apply.  The **genus** is still reported: it does not depend
    /// on the model.
    RealModel {
        /// `deg a`, always even here.
        degree: usize,
        /// The genus, which *is* known.
        genus: usize,
        /// The operation that refused.
        operation: &'static str,
    },
    /// The computation needs every place in play to be ℚ-**rational**, and one
    /// is not — or could not be shown to be.
    ///
    /// [`super::Place`] represents a degree-1 place `(α, β)` with `α, β ∈ ℚ`,
    /// plus the place at infinity.  A conjugate pair `(α, ±√c)` with `c` a
    /// non-square, or an `x`-coordinate that is not rational at all, is a place
    /// of degree 2 or more and has no representation here.  Silently dropping
    /// it would return a divisor of the wrong degree, so it is a refusal.
    NonRationalSupport {
        /// What produced the non-rational place.
        context: String,
    },
    /// A place `(α, β)` was supplied with `β² ≠ a(α)`: it is not on the curve.
    PlaceNotOnCurve {
        /// Rendered `x`-coordinate.
        x: String,
        /// Rendered `y`-coordinate.
        y: String,
    },
    /// A divisor-class operation was handed a divisor whose degree is not zero.
    ///
    /// The group modelled here is `Pic⁰`, the classes of **degree-zero**
    /// divisors.  A divisor of degree `n ≠ 0` has no class in it; subtract
    /// `n·∞` first if that is what you meant.
    NotDegreeZero {
        /// The degree that was found.
        degree: String,
    },
    /// The order of a torsion class could not be pinned down within scope.
    ///
    /// The reduction-modulo-good-primes test needs at least two good primes
    /// below its search ceiling, and the reconstructed candidate must fit in a
    /// machine word.  Neither a torsion certificate nor a non-torsion one.
    UndecidedOrder {
        /// What ran out.
        reason: String,
    },
    /// The class is **not** torsion: it has infinite order.
    ///
    /// **A verdict, not a refusal.** Reduction modulo good primes is injective
    /// on prime-to-`p` torsion, so orders that disagree across two good primes
    /// prove the class is of infinite order; so does an exact ℚ-Cantor test
    /// that rejects the only candidate order the reconstruction allows.
    /// Retrying with a bigger budget will not change this answer.
    NonTorsion,
    /// `div(0)` was requested.  The zero function has no divisor.
    ZeroFunction,
    /// Two objects built over different function fields were combined.
    CurveMismatch {
        /// Rendered left curve.
        left: String,
        /// Rendered right curve.
        right: String,
    },
    /// A multiplicity, degree or order is too large for this implementation.
    ///
    /// Cantor arithmetic here folds a place in by repeated addition bounded by
    /// a machine word, and the Riemann–Roch solver builds a dense matrix.
    TooLarge {
        /// What exceeded what.
        reason: String,
    },
    /// A result was **computed and then withheld** because it failed its own
    /// consistency check.
    ///
    /// Two checks can fire.  `divisor_of_function` cross-checks the pole order
    /// at infinity obtained from degree balance against the one read off the
    /// degrees of `p` and `q`; `riemann_roch` checks the dimension it found
    /// against Riemann's inequality `dim ≥ deg D + 1 − g` and, above the
    /// canonical degree, against the Riemann–Roch equality itself.  A
    /// disagreement means this module has a bug, and returning the answer
    /// anyway is the silent-wrong-answer failure mode the crate refuses by
    /// policy.
    SelfCheckFailed {
        /// Which check, and what disagreed.
        detail: String,
    },
}

impl fmt::Display for FunctionFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionFieldError::UnsupportedModel { reason } => write!(
                f,
                "not a supported curve model: {reason}. This module implements \
                 ℚ(x)[y]/(y² − a(x)) with a squarefree of degree ≥ 1"
            ),
            FunctionFieldError::RealModel {
                degree,
                genus,
                operation,
            } => write!(
                f,
                "`{operation}` is implemented only for the imaginary (odd-degree) \
                 hyperelliptic model; `deg a = {degree}` is even, so the curve has two \
                 places above x = ∞ and no single rational base point (the genus, {genus}, \
                 is still available)"
            ),
            FunctionFieldError::NonRationalSupport { context } => write!(
                f,
                "a place of degree > 1 appears ({context}); only ℚ-rational places \
                 (α, β) and the place at infinity can be represented here"
            ),
            FunctionFieldError::PlaceNotOnCurve { x, y } => {
                write!(f, "({x}, {y}) is not on the curve: y² ≠ a(x)")
            }
            FunctionFieldError::NotDegreeZero { degree } => write!(
                f,
                "divisor has degree {degree}, but the class group modelled here is Pic⁰; \
                 subtract deg(D)·∞ to land in it"
            ),
            FunctionFieldError::UndecidedOrder { reason } => write!(
                f,
                "could not determine the order of the class: {reason}. This is neither a \
                 torsion nor a non-torsion certificate"
            ),
            FunctionFieldError::NonTorsion => write!(
                f,
                "the divisor class is not torsion — it has infinite order, so no multiple \
                 of the divisor is principal"
            ),
            FunctionFieldError::ZeroFunction => {
                write!(f, "the zero function has no divisor")
            }
            FunctionFieldError::CurveMismatch { left, right } => write!(
                f,
                "these objects live on different curves ({left} and {right}); divisors and \
                 classes can only be combined within one function field"
            ),
            FunctionFieldError::TooLarge { reason } => {
                write!(f, "too large for this implementation: {reason}")
            }
            FunctionFieldError::SelfCheckFailed { detail } => write!(
                f,
                "a result was computed and then withheld because it failed its own \
                 consistency check: {detail}"
            ),
        }
    }
}

impl std::error::Error for FunctionFieldError {}

impl crate::errors::AlkahestError for FunctionFieldError {
    fn code(&self) -> &'static str {
        match self {
            FunctionFieldError::UnsupportedModel { .. } => "E-FFLD-001",
            FunctionFieldError::RealModel { .. } => "E-FFLD-002",
            FunctionFieldError::NonRationalSupport { .. } => "E-FFLD-003",
            FunctionFieldError::PlaceNotOnCurve { .. } => "E-FFLD-004",
            FunctionFieldError::NotDegreeZero { .. } => "E-FFLD-005",
            FunctionFieldError::UndecidedOrder { .. } => "E-FFLD-006",
            FunctionFieldError::NonTorsion => "E-FFLD-007",
            FunctionFieldError::ZeroFunction => "E-FFLD-008",
            FunctionFieldError::CurveMismatch { .. } => "E-FFLD-009",
            FunctionFieldError::TooLarge { .. } => "E-FFLD-010",
            FunctionFieldError::SelfCheckFailed { .. } => "E-FFLD-011",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            FunctionFieldError::UnsupportedModel { .. } => Some(
                "supply c₂·y² + c₁(x)·y + c₀(x) with c₂ a non-zero rational constant, whose discriminant c₁² − 4c₂c₀ is non-constant; higher-degree plane curves are not modelled",
            ),
            FunctionFieldError::RealModel { .. } => Some(
                "move the model to odd degree by sending a rational root of a(x) to infinity, or use genus() alone, which is model-independent",
            ),
            FunctionFieldError::NonRationalSupport { .. } => Some(
                "restrict to divisors supported on rational places, or work with the divisor class (Mumford form), which represents conjugate places implicitly",
            ),
            FunctionFieldError::PlaceNotOnCurve { .. } => Some(
                "check the sign and the model: the place must satisfy y² = a(x) in the *normalised* coordinates reported by FunctionField::curve()",
            ),
            FunctionFieldError::NotDegreeZero { .. } => {
                Some("replace D by D − deg(D)·∞ before asking for its class")
            }
            FunctionFieldError::UndecidedOrder { .. } => Some(
                "record the result as undecided — never as non-torsion; a larger prime search or a different model may settle it",
            ),
            FunctionFieldError::NonTorsion => Some(
                "this is a verdict: no multiple of the divisor is principal, so stop looking for one",
            ),
            FunctionFieldError::ZeroFunction => {
                Some("pass a non-zero function; div(0) is not defined")
            }
            FunctionFieldError::CurveMismatch { .. } => {
                Some("rebuild both operands over the same FunctionField")
            }
            FunctionFieldError::TooLarge { .. } => Some(
                "reduce the multiplicities or the degree of the divisor; the Cantor and linear-algebra steps here are bounded by a machine word",
            ),
            FunctionFieldError::SelfCheckFailed { .. } => Some(
                "report this as a bug with the curve and divisor that produced it; the answer was withheld rather than returned wrong",
            ),
        }
    }
}
