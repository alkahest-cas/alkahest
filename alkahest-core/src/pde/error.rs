//! `E-PDE-001 … E-PDE-049`: the errors of the PDE foundation layer (milestone
//! M0 — jets, problem objects, classification, verification, the `Sum` /
//! `Indexed` / `RootOfFamily` heads and derivative-aware transforms).
//!
//! Later milestones own their own ranges (M1 `050–099`, M2 `100–149`, …); this
//! enum is `#[non_exhaustive]` so they can add variants without a semver break.

use std::fmt;

use crate::errors::AlkahestError;

/// Failure of a PDE-layer operation.  Every variant carries a human-readable
/// detail naming the offending expression; branch on [`AlkahestError::code`],
/// never on the prose.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PdeError {
    /// `E-PDE-001` — an unknown is not an undefined function applied to
    /// distinct plain symbols (`u(x, t)`).
    NotAnUnknown(String),
    /// `E-PDE-002` — the unknowns and independent variables are inconsistent:
    /// an argument that is not an independent variable, a repeated variable or
    /// unknown, or no independent variables at all.
    InconsistentVariables(String),
    /// `E-PDE-003` — an unknown occurs evaluated somewhere other than at its
    /// declared arguments (`u(x, 0)`, `u(2x, t)`), so it has no jet coordinate.
    OffJet(String),
    /// `E-PDE-004` — a derivative is of higher order than the jet space holds.
    OrderExceeded(String),
    /// `E-PDE-005` — a jet coordinate's name is already taken by a symbol of
    /// the input, so the jet ↔ derivative map would not be a bijection.
    JetNameCollision(String),
    /// `E-PDE-006` — the system is not polynomial / rational in the jet
    /// coordinates (`sin(u_x)`), so it has no Gröbner or Rosenfeld–Gröbner form.
    NotRationalInJets(String),
    /// `E-PDE-007` — an equation mentions no unknown (or no derivative of
    /// one): it is not a differential equation in the declared unknowns.
    NotADifferentialEquation(String),
    /// `E-PDE-008` — an empty system, or an operation that needs exactly one
    /// equation / unknown was given several.
    SystemShape(String),
    /// `E-PDE-009` — an ODE naming convention (`y'`, `q_dot`, `dy/dt`) was
    /// requested for a jet space with more than one independent variable.
    NamingNeedsOneVariable(String),
    /// `E-PDE-010` — a malformed domain (empty interval, wrong dimension,
    /// non-positive radius, a box whose arity does not match its variables).
    InvalidDomain(String),
    /// `E-PDE-011` — a malformed boundary or initial condition (it mentions no
    /// unknown).
    InvalidCondition(String),
    /// `E-PDE-012` — classification is not defined for this input (a system,
    /// or an equation whose highest-order part vanishes identically).
    ClassificationUnsupported(String),
    /// `E-PDE-013` — a candidate solution is malformed for verification (the
    /// wrong number of components for the unknowns).
    InvalidCandidate(String),
    /// `E-PDE-020` — a malformed `Sum(body, k, lo, hi)` node: the bound
    /// variable is not a symbol, or a bound mentions the bound variable.
    MalformedSum(String),
    /// `E-PDE-021` — a `Sum` cannot be expanded or evaluated: a bound is not
    /// an integer after substitution (symbolic `d`, `∞`), or the range is
    /// larger than the term cap.
    SumNotFinite(String),
    /// `E-PDE-022` — a `Sum` cannot be differentiated term-wise: a bound
    /// depends on the differentiation variable.
    SumDifferentiation(String),
    /// `E-PDE-023` — a malformed `Indexed(x, i)` node (the base is not a
    /// symbol).
    MalformedIndexed(String),
    /// `E-PDE-030` — a malformed `RootOfFamily(F, λ, n, branch)` node.
    MalformedRootFamily(String),
    /// `E-PDE-031` — the requested root of a `RootOfFamily` could not be
    /// certified: no sign change was isolated in its bracket, the bracket holds
    /// more than one root, or `F` cannot be evaluated in ball arithmetic.
    RootFamilyNotCertified(String),
    /// `E-PDE-040` — a transform cannot act on an unknown: the transform
    /// variable occurs in an argument other than as a bare slot (`u(t²)`,
    /// `u(x, 2t)`), or in more than one slot.
    TransformUnknown(String),
}

impl fmt::Display for PdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (what, d) = match self {
            PdeError::NotAnUnknown(d) => ("not an unknown function", d),
            PdeError::InconsistentVariables(d) => ("inconsistent variables", d),
            PdeError::OffJet(d) => ("unknown evaluated off its jet", d),
            PdeError::OrderExceeded(d) => ("derivative order exceeds the jet space", d),
            PdeError::JetNameCollision(d) => ("jet name collision", d),
            PdeError::NotRationalInJets(d) => ("not rational in the jet coordinates", d),
            PdeError::NotADifferentialEquation(d) => ("not a differential equation", d),
            PdeError::SystemShape(d) => ("system shape", d),
            PdeError::NamingNeedsOneVariable(d) => ("ODE naming needs one variable", d),
            PdeError::InvalidDomain(d) => ("invalid domain", d),
            PdeError::InvalidCondition(d) => ("invalid condition", d),
            PdeError::ClassificationUnsupported(d) => ("classification unsupported", d),
            PdeError::InvalidCandidate(d) => ("invalid candidate solution", d),
            PdeError::MalformedSum(d) => ("malformed Sum", d),
            PdeError::SumNotFinite(d) => ("Sum is not finite", d),
            PdeError::SumDifferentiation(d) => ("Sum cannot be differentiated term-wise", d),
            PdeError::MalformedIndexed(d) => ("malformed Indexed", d),
            PdeError::MalformedRootFamily(d) => ("malformed RootOfFamily", d),
            PdeError::RootFamilyNotCertified(d) => ("RootOfFamily root not certified", d),
            PdeError::TransformUnknown(d) => ("transform cannot act on the unknown", d),
        };
        write!(f, "{what}: {d}")
    }
}

impl std::error::Error for PdeError {}

impl AlkahestError for PdeError {
    fn code(&self) -> &'static str {
        match self {
            PdeError::NotAnUnknown(_) => "E-PDE-001",
            PdeError::InconsistentVariables(_) => "E-PDE-002",
            PdeError::OffJet(_) => "E-PDE-003",
            PdeError::OrderExceeded(_) => "E-PDE-004",
            PdeError::JetNameCollision(_) => "E-PDE-005",
            PdeError::NotRationalInJets(_) => "E-PDE-006",
            PdeError::NotADifferentialEquation(_) => "E-PDE-007",
            PdeError::SystemShape(_) => "E-PDE-008",
            PdeError::NamingNeedsOneVariable(_) => "E-PDE-009",
            PdeError::InvalidDomain(_) => "E-PDE-010",
            PdeError::InvalidCondition(_) => "E-PDE-011",
            PdeError::ClassificationUnsupported(_) => "E-PDE-012",
            PdeError::InvalidCandidate(_) => "E-PDE-013",
            PdeError::MalformedSum(_) => "E-PDE-020",
            PdeError::SumNotFinite(_) => "E-PDE-021",
            PdeError::SumDifferentiation(_) => "E-PDE-022",
            PdeError::MalformedIndexed(_) => "E-PDE-023",
            PdeError::MalformedRootFamily(_) => "E-PDE-030",
            PdeError::RootFamilyNotCertified(_) => "E-PDE-031",
            PdeError::TransformUnknown(_) => "E-PDE-040",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        crate::errors::codes::REGISTRY
            .iter()
            .find(|s| s.code == self.code())
            .and_then(|s| s.remediation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_code_is_registered_with_a_remediation() {
        let all = [
            PdeError::NotAnUnknown(String::new()),
            PdeError::InconsistentVariables(String::new()),
            PdeError::OffJet(String::new()),
            PdeError::OrderExceeded(String::new()),
            PdeError::JetNameCollision(String::new()),
            PdeError::NotRationalInJets(String::new()),
            PdeError::NotADifferentialEquation(String::new()),
            PdeError::SystemShape(String::new()),
            PdeError::NamingNeedsOneVariable(String::new()),
            PdeError::InvalidDomain(String::new()),
            PdeError::InvalidCondition(String::new()),
            PdeError::ClassificationUnsupported(String::new()),
            PdeError::InvalidCandidate(String::new()),
            PdeError::MalformedSum(String::new()),
            PdeError::SumNotFinite(String::new()),
            PdeError::SumDifferentiation(String::new()),
            PdeError::MalformedIndexed(String::new()),
            PdeError::MalformedRootFamily(String::new()),
            PdeError::RootFamilyNotCertified(String::new()),
            PdeError::TransformUnknown(String::new()),
        ];
        for e in all {
            let code = e.code();
            assert!(code.starts_with("E-PDE-0"), "{code}");
            let n: u32 = code[6..].parse().unwrap();
            assert!((1..=49).contains(&n), "{code} outside the M0 range");
            assert!(e.remediation().is_some(), "{code} not registered");
        }
    }
}
