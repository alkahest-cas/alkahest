//! Refusals for exact linear and mixed-integer programming.
//!
//! Every variant is a refusal, never a fallback. In particular
//! [`LinprogError::NodeLimit`] and [`LinprogError::Budget`] are *not* an
//! answer: the incumbent they carry is the best integer point found so far,
//! reported together with the best open bound, and nothing claims it is
//! optimal.

use std::fmt;

use rug::Rational;

use crate::budget::BudgetError;

/// The best integer-feasible point a branch-and-bound search had found when it
/// stopped, with the objective value it attains (in the caller's sense).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incumbent {
    /// The point, one entry per decision variable.
    pub x: Vec<Rational>,
    /// Its objective value `c · x`.
    pub objective: Rational,
}

/// Why a linear or mixed-integer programme refused.
///
/// Codes are `E-LP-001` … `E-LP-006`, plus the `E-BUDGET-*` code of the
/// budget trip carried by [`LinprogError::Budget`]; see
/// [`crate::errors::codes::REGISTRY`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinprogError {
    /// The data do not describe a programme: a row of the wrong length, a
    /// right-hand side whose length differs from the number of rows, a bound
    /// list of the wrong length, or an integrality list of the wrong length.
    Shape {
        /// What is wrong, naming the offending argument.
        detail: String,
    },
    /// A coefficient is not an exact rational (a float, or an irrational
    /// constant such as `sqrt(2)`).
    ///
    /// The solver is exact; rounding a float to the nearest rational would
    /// make every certificate it returns a statement about a different
    /// programme than the one the caller meant.
    Inexact {
        /// Which coefficient, and what it was.
        detail: String,
    },
    /// An expression handed to the symbolic front end is not affine in the
    /// decision variables, or mentions a symbol that is not one of them.
    NonLinear {
        /// The offending expression and why it is not affine.
        detail: String,
    },
    /// A constraint handed to the symbolic front end is not `≤`, `≥` or `=`
    /// (e.g. a strict inequality, `≠`, or a non-relational expression).
    ///
    /// A strict inequality has no optimum in general (`min x s.t. x > 0`), so
    /// it is refused rather than silently relaxed to `≤`.
    UnsupportedConstraint {
        /// The constraint and why it was refused.
        detail: String,
    },
    /// Branch and bound explored `max_nodes` nodes without closing the gap.
    ///
    /// The incumbent (if any) is integer-feasible but **not** proven optimal;
    /// `bound` is the best LP bound over the nodes still open, so the optimum
    /// lies between `bound` and the incumbent's objective.
    NodeLimit {
        /// The node cap that was reached.
        nodes: usize,
        /// The best integer point found, if any.
        incumbent: Option<Incumbent>,
        /// The best bound over the open nodes, in the caller's sense.
        bound: Option<Rational>,
    },
    /// The active [`crate::budget::Budget`] tripped (wall clock, steps or
    /// cancellation). The code is the budget's own `E-BUDGET-*` code.
    Budget {
        /// The budget trip.
        source: BudgetError,
        /// The best integer point found before the trip (MILP only).
        incumbent: Option<Incumbent>,
        /// The best open bound before the trip (MILP only).
        bound: Option<Rational>,
    },
    /// A certificate computed by the simplex failed its own exact check.
    ///
    /// This is a bug, and the result is withheld rather than returned: an
    /// "optimal" point whose dual does not certify it is a confident wrong
    /// answer.
    Internal {
        /// Which check failed.
        detail: String,
    },
}

impl fmt::Display for LinprogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LinprogError::Shape { detail } => {
                write!(f, "the programme is malformed: {detail}")
            }
            LinprogError::Inexact { detail } => write!(
                f,
                "linear programming is exact here, and a coefficient is not an exact \
                 rational: {detail}"
            ),
            LinprogError::NonLinear { detail } => {
                write!(
                    f,
                    "expression is not affine in the decision variables: {detail}"
                )
            }
            LinprogError::UnsupportedConstraint { detail } => write!(
                f,
                "constraint must be a relation <=, >= or == between affine expressions: {detail}"
            ),
            LinprogError::NodeLimit {
                nodes,
                incumbent,
                bound,
            } => {
                write!(
                    f,
                    "branch and bound stopped after {nodes} nodes without proving optimality"
                )?;
                match incumbent {
                    Some(inc) => write!(f, "; best integer objective found {}", inc.objective)?,
                    None => write!(f, "; no integer-feasible point found")?,
                }
                if let Some(b) = bound {
                    write!(f, "; best open bound {b}")?;
                }
                Ok(())
            }
            LinprogError::Budget {
                source, incumbent, ..
            } => {
                write!(f, "linear programming stopped: {source}")?;
                if let Some(inc) = incumbent {
                    write!(f, "; best integer objective found {}", inc.objective)?;
                }
                Ok(())
            }
            LinprogError::Internal { detail } => write!(
                f,
                "internal error: a simplex certificate failed its exact check ({detail}); \
                 the result was withheld"
            ),
        }
    }
}

impl std::error::Error for LinprogError {}

impl crate::errors::AlkahestError for LinprogError {
    fn code(&self) -> &'static str {
        match self {
            LinprogError::Shape { .. } => "E-LP-001",
            LinprogError::Inexact { .. } => "E-LP-002",
            LinprogError::NonLinear { .. } => "E-LP-003",
            LinprogError::UnsupportedConstraint { .. } => "E-LP-004",
            LinprogError::NodeLimit { .. } => "E-LP-005",
            LinprogError::Internal { .. } => "E-LP-006",
            LinprogError::Budget { source, .. } => crate::errors::AlkahestError::code(source),
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        if let LinprogError::Budget { source, .. } = self {
            return crate::errors::AlkahestError::remediation(source);
        }
        crate::errors::codes::REGISTRY
            .iter()
            .find(|spec| spec.code == crate::errors::AlkahestError::code(self))
            .and_then(|spec| spec.remediation)
    }
}

impl From<BudgetError> for LinprogError {
    fn from(source: BudgetError) -> Self {
        LinprogError::Budget {
            source,
            incumbent: None,
            bound: None,
        }
    }
}
