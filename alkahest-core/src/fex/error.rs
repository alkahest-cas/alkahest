//! Refusals raised by the FEX evaluation engine (`E-FEX-001` … `E-FEX-008`).
//!
//! Every variant names what the caller passed and what was expected. None of
//! them is a numerical failure of a candidate: a candidate whose value
//! overflows, divides by zero or leaves the domain of `log` is scored with
//! loss `+∞`, not refused, so a search can keep going.

use std::fmt;

use crate::errors::AlkahestError;

/// Why a FEX template, problem, evaluator or fit refused.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FexError {
    /// `E-FEX-001`: the tree-shape specification does not parse, is empty, or
    /// is too large.
    Shape {
        /// What is wrong with the specification.
        detail: String,
    },
    /// `E-FEX-002`: an operator name is not in the unary or binary table.
    UnknownOperator {
        /// The name that was not recognised.
        name: String,
        /// `"unary"` or `"binary"`.
        table: &'static str,
    },
    /// `E-FEX-003`: an opcode vector has the wrong length, or an opcode is out
    /// of range for its slot.
    Opcode {
        /// What is wrong with the opcode vector.
        detail: String,
    },
    /// `E-FEX-004`: a parameter vector has the wrong length.
    Params {
        /// Expected parameter count.
        expected: usize,
        /// The length that was passed.
        got: usize,
    },
    /// `E-FEX-005`: a point array does not have one column per coordinate, or
    /// a batch has inconsistent shapes.
    Points {
        /// What is wrong with the shapes.
        detail: String,
    },
    /// `E-FEX-006`: the PDE operator (or a source term) mentions a symbol that
    /// is neither a jet symbol nor a coordinate, or cannot be differentiated
    /// or compiled.
    Operator {
        /// Why the expression was refused.
        detail: String,
    },
    /// `E-FEX-007`: the template and the problem disagree on the dimension or
    /// the time coordinate is out of range.
    Dimension {
        /// What disagrees.
        detail: String,
    },
    /// `E-FEX-008`: the requested quantity does not exist for this loss — a
    /// residual vector (and so Levenberg–Marquardt) for a Rayleigh-quotient or
    /// normalised loss.
    LossKind {
        /// What was asked for and why it is not available.
        detail: String,
    },
}

impl fmt::Display for FexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FexError::Shape { detail } => write!(f, "invalid FEX tree shape: {detail}"),
            FexError::UnknownOperator { name, table } => {
                write!(f, "unknown FEX {table} operator {name:?}")
            }
            FexError::Opcode { detail } => write!(f, "invalid FEX opcode vector: {detail}"),
            FexError::Params { expected, got } => write!(
                f,
                "FEX parameter vector has length {got}, the template needs {expected}"
            ),
            FexError::Points { detail } => write!(f, "invalid FEX point array: {detail}"),
            FexError::Operator { detail } => write!(f, "unsupported FEX operator: {detail}"),
            FexError::Dimension { detail } => write!(f, "FEX dimension mismatch: {detail}"),
            FexError::LossKind { detail } => write!(f, "FEX loss kind: {detail}"),
        }
    }
}

impl std::error::Error for FexError {}

impl AlkahestError for FexError {
    fn code(&self) -> &'static str {
        match self {
            FexError::Shape { .. } => "E-FEX-001",
            FexError::UnknownOperator { .. } => "E-FEX-002",
            FexError::Opcode { .. } => "E-FEX-003",
            FexError::Params { .. } => "E-FEX-004",
            FexError::Points { .. } => "E-FEX-005",
            FexError::Operator { .. } => "E-FEX-006",
            FexError::Dimension { .. } => "E-FEX-007",
            FexError::LossKind { .. } => "E-FEX-008",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            FexError::Shape { .. } => {
                "write the shape with L (leaf), U(child) and B(left,right), e.g. \"U(B(L,L))\", or use a preset: depth1, depth2, depth2_sub, depth2_rml, depth3"
            }
            FexError::UnknownOperator { .. } => {
                "unary operators: 0 1 id sq cube quartic exp sin cos log abs; binary: + - * /"
            }
            FexError::Opcode { .. } => {
                "pass one opcode per slot (Template.n_slots), each below the size of that slot's table (unary or binary)"
            }
            FexError::Params { .. } => {
                "pass Template.n_params parameters: d+1 per leaf (weights, then bias), 2 per inner unary slot (alpha, beta), none per binary slot"
            }
            FexError::Points { .. } => {
                "pass C-contiguous float64 arrays of shape (n_points, dim); a batch of opcodes is (batch, n_slots) and of parameters (batch, n_params)"
            }
            FexError::Operator { .. } => {
                "build the operator from the Jet symbols (u, lap, grad[i], ut) and the coordinates only; source terms may use the coordinates only"
            }
            FexError::Dimension { .. } => {
                "build the template and the problem with the same dim, and give the time coordinate as an index below dim"
            }
            FexError::LossKind { .. } => {
                "use loss_and_grad (and the adam or bfgs fitters) for Rayleigh-quotient and normalised losses; residual_and_jacobian and Levenberg-Marquardt need a least-squares problem"
            }
        })
    }
}
