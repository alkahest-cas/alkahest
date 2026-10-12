//! Errors raised by the lowering stages.
//!
//! A lowering stage either produces MLIR that computes exactly the source
//! expression or says why it cannot.  It never substitutes a placeholder
//! (`0.0`, an undeclared `func.call`) for something it could not lower: the
//! text goes to another toolchain, which will not compare it back.

use alkahest_core::stablehlo::StableHloError;
use alkahest_core::AlkahestError;
use std::fmt;

/// Why a lowering stage refused an expression.
///
/// | Variant | Code |
/// |---|---|
/// | [`UnboundSymbol`](Self::UnboundSymbol) | `E-MLIR-001` |
/// | [`UnsupportedFunction`](Self::UnsupportedFunction) | `E-MLIR-002` |
/// | [`UnsupportedNode`](Self::UnsupportedNode) | `E-MLIR-003` |
/// | [`NonFiniteConstant`](Self::NonFiniteConstant) | `E-MLIR-004` |
/// | [`StableHlo`](Self::StableHlo) | the wrapped `E-STABLEHLO-NNN` |
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum LoweringError {
    /// A symbol that is neither one of the function's inputs nor `pi`.
    UnboundSymbol(String),
    /// A function with no exact lowering at this stage (`erf`, `gamma`, a user
    /// function, `tan` in the LLVM dialect …), or a known one at the wrong arity.
    UnsupportedFunction {
        /// The function's name.
        name: String,
        /// The number of arguments it was applied to.
        arity: usize,
    },
    /// A node with no numeric value (`O(x)`, a quantifier, a bare predicate)
    /// or a malformed `Piecewise` condition.
    UnsupportedNode(String),
    /// A `NaN` or infinite constant, which has no portable literal.
    NonFiniteConstant(f64),
    /// The StableHLO stage refused; carries its own `E-STABLEHLO-NNN` code.
    StableHlo(StableHloError),
}

impl fmt::Display for LoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoweringError::UnboundSymbol(name) => write!(
                f,
                "MLIR lowering: symbol `{name}` is not among the function inputs; \
                 no module was emitted"
            ),
            LoweringError::UnsupportedFunction { name, arity } => write!(
                f,
                "MLIR lowering: `{name}` applied to {arity} argument(s) has no exact \
                 lowering at this stage; no module was emitted"
            ),
            LoweringError::UnsupportedNode(what) => write!(
                f,
                "MLIR lowering: {what} cannot be lowered to straight-line numeric code"
            ),
            LoweringError::NonFiniteConstant(v) => write!(
                f,
                "MLIR lowering: the non-finite constant {v} has no MLIR float literal"
            ),
            LoweringError::StableHlo(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for LoweringError {}

impl From<StableHloError> for LoweringError {
    fn from(e: StableHloError) -> Self {
        LoweringError::StableHlo(e)
    }
}

impl AlkahestError for LoweringError {
    fn code(&self) -> &'static str {
        match self {
            LoweringError::UnboundSymbol(_) => "E-MLIR-001",
            LoweringError::UnsupportedFunction { .. } => "E-MLIR-002",
            LoweringError::UnsupportedNode(_) => "E-MLIR-003",
            LoweringError::NonFiniteConstant(_) => "E-MLIR-004",
            LoweringError::StableHlo(e) => e.code(),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            LoweringError::UnboundSymbol(_) => {
                Some("add the symbol to the function's inputs, or substitute a value for it")
            }
            LoweringError::UnsupportedFunction { .. } => Some(
                "rewrite the function in terms of sin, cos, exp, log, sqrt, abs, or \
                 substitute it before lowering",
            ),
            LoweringError::UnsupportedNode(_) => None,
            LoweringError::NonFiniteConstant(_) => None,
            LoweringError::StableHlo(e) => e.remediation(),
        }
    }
}
