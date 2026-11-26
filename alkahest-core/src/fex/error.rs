//! Refusals raised by the FEX evaluation engine (`E-FEX-001` … `E-FEX-008`)
//! and by finalisation (`E-FEX-009` … `E-FEX-011`, raised from Python's
//! `alkahest.fex.finalize` / `recovery_metric`), and by the F4 search layer
//! (`E-FEX-012` … `E-FEX-014`, raised from Python's `analyze` /
//! `transform_problem` / `reduce_problem`, `search` / `solve` and
//! `TorchEvaluator`).
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
    /// `E-FEX-009`: finalisation was given a candidate without its template,
    /// or a candidate and a problem that do not belong together.
    Finalize {
        /// What is missing or inconsistent.
        detail: String,
    },
    /// `E-FEX-010`: a symbolic-`d` problem family does not reduce to the
    /// given problem at the problem's own dimension.
    Family {
        /// Which part of the family disagrees.
        detail: String,
    },
    /// `E-FEX-011`: the recovery metric was given something other than two
    /// expressions.
    Recovery {
        /// What was passed.
        detail: String,
    },
    /// `E-FEX-012`: a dependent-variable substitution or a variable
    /// reduction was asked of a problem it does not apply to.
    Transform {
        /// Why it does not apply.
        detail: String,
    },
    /// `E-FEX-013`: malformed search-controller options (slot restrictions,
    /// baseline, population, priority queue, curriculum).
    SearchOptions {
        /// Which option and why.
        detail: String,
    },
    /// `E-FEX-014`: the GPU (PyTorch) evaluator is unavailable or does not
    /// cover the problem.
    Backend {
        /// What is missing.
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
            FexError::Finalize { detail } => write!(f, "FEX finalize: {detail}"),
            FexError::Family { detail } => write!(f, "FEX problem family: {detail}"),
            FexError::Recovery { detail } => write!(f, "FEX recovery metric: {detail}"),
            FexError::Transform { detail } => write!(f, "FEX transform: {detail}"),
            FexError::SearchOptions { detail } => write!(f, "FEX search options: {detail}"),
            FexError::Backend { detail } => write!(f, "FEX evaluator backend: {detail}"),
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
            FexError::Finalize { .. } => "E-FEX-009",
            FexError::Family { .. } => "E-FEX-010",
            FexError::Recovery { .. } => "E-FEX-011",
            FexError::Transform { .. } => "E-FEX-012",
            FexError::SearchOptions { .. } => "E-FEX-013",
            FexError::Backend { .. } => "E-FEX-014",
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
            FexError::Finalize { .. } => {
                "pass the SearchResult returned by fex.search, or a FitResult together with template=; the problem must be the fex.Problem the candidate was fitted to"
            }
            FexError::Family { .. } => {
                "build the family from fex.family_symbols(problem): rhs / bc / potential as expressions in d, Indexed(x, i) and t that reduce to the problem's own source terms at the problem's dimension"
            }
            FexError::Recovery { .. } => {
                "pass two Expr from the same ExprPool (or a SearchResult / finalize result for 'found')"
            }
            FexError::Transform { .. } => {
                "propose only the transforms analyze() lists for this problem: u = e^v needs an operator that is homogeneous in u (rhs = 0) or an eigenproblem, and boundary data that are positive or absent; a variable reduction needs operator and data free of the dropped coordinates once their derivatives vanish"
            }
            FexError::SearchOptions { .. } => {
                "give slot_ops as {'leaf' | 'unary' | 'binary': [operator names]} or one list of names per slot, naming operators of the template's tables and leaving every slot at least one; baseline is 'quantile', 'ewma' or 'mean'; population, pqt_size and curriculum are non-negative integers"
            }
            FexError::Backend { .. } => {
                "install PyTorch (with CUDA for device='cuda'), and use the GPU evaluator for the built-in problems (poisson, transport, cubic_schrodinger, eigen); operators written as expressions and the centred loss run on alkahest.fex.Evaluator"
            }
        })
    }
}
