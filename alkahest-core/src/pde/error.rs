//! `E-PDE-001 … E-PDE-049`: the errors of the PDE foundation layer (milestone
//! M0 — jets, problem objects, classification, verification, the `Sum` /
//! `Indexed` / `RootOfFamily` heads and derivative-aware transforms).
//!
//! `E-PDE-050 … E-PDE-099` belong to `pdsolve` (milestone M1, first order).
//!
//! Later milestones own their own ranges (M2 `100–149`, …); this
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
    /// `E-PDE-050` — `pdsolve` has no method for this problem: a system, an
    /// equation of order other than one (until later milestones add tiers),
    /// a single independent variable, or an unknown hint name.
    PdsolveUnsupported(String),
    /// `E-PDE-051` — every applicable method was tried and none produced an
    /// answer that verified exactly; candidates that failed verification are
    /// discarded, never returned.
    NoVerifiedSolution(String),
    /// `E-PDE-052` — the Cauchy data lie on a characteristic curve (the
    /// non-characteristic Jacobian vanishes identically), so the data do not
    /// determine a solution.
    CharacteristicData(String),
    /// `E-PDE-053` — the side conditions are not Cauchy data `pdsolve` can
    /// impose: a condition on a derivative, data on a manifold of the wrong
    /// dimension, or parameters that could not be eliminated.
    CauchyUnsupported(String),
    /// `E-PDE-054` — a malformed `Integral(f, t, lo, hi)` / `Integral(f, x)`
    /// node: the integration variable is not a symbol, or a bound mentions it.
    MalformedIntegral(String),
    /// `E-PDE-100` — the second-order solvers need a linear equation of
    /// order two in exactly two independent variables.
    NotSecondOrderLinear2D(String),
    /// `E-PDE-101` — no first integral of a characteristic ODE was found in
    /// closed form, or the characteristic coordinates did not reduce the
    /// principal part.
    CharacteristicsNotIntegrated(String),
    /// `E-PDE-102` — the type is not the same on the whole plane (mixed or
    /// undetermined): a canonical form exists only region by region.
    TypeNotUniform(String),
    /// `E-PDE-103` — the Laplace cascade spent its step budget without an
    /// invariant vanishing (termination has no a priori bound).
    LaplaceBudgetExhausted(String),
    /// `E-PDE-104` — no implemented method gives a closed-form solution.
    NoClosedForm(String),
    /// `E-PDE-105` — a candidate solution was produced but substitution did
    /// not prove it; it is withheld.
    SecondOrderNotVerified(String),
    /// `E-PDE-106` — the solution would need a quadrature of an arbitrary
    /// function (or of a coefficient) that has no closed form.
    QuadratureOfArbitraryFunction(String),
    /// `E-PDE-107` — the conditions are not Cauchy data the second-order
    /// solver handles (`u(x, y₀) = f`, `u_y(x, y₀) = g`), or the data line is
    /// characteristic, or `∫ g` does not close.
    CauchyDataUnsupported(String),
    /// `E-PDE-108` — the Laplace cascade needs `u_xy + a u_x + b u_y + c u = s`
    /// (principal part a multiple of `u_xy`).
    NotLaplaceForm(String),
    /// `E-PDE-150` — an eigenproblem outside the implemented classes: `dsolve` does not solve the
    /// separated ODE, periodic conditions with non-constant coefficients, or a
    /// singular endpoint of an unrecognised equation.
    SlUnsupported(String),
    /// `E-PDE-151` — a malformed eigenproblem boundary condition (`α = β = 0` in
    /// `αX + βX' = 0`, a `bounded` / `decay` condition at a regular end, an
    /// end that is not on the interval).
    SlInvalidBoundary(String),
    /// `E-PDE-152` — the eigenvalue condition vanishes identically, so every `λ` would be
    /// an eigenvalue: the boundary conditions do not determine a self-adjoint
    /// problem.
    SlDegenerate(String),
    /// `E-PDE-153` — an eigenpair failed verification: the ODE or boundary residual
    /// was refuted.  Nothing is returned rather than an unverified pair.
    SlVerificationFailed(String),
    /// `E-PDE-160` — the transform method does not apply: not one linear,
    /// constant-coefficient, homogeneous PDE in two variables, or a domain the
    /// transform does not match (Fourier needs a line, Laplace a half-line).
    TransformMethodUnsupported(String),
    /// `E-PDE-161` — initial or boundary data is missing, duplicated or of the wrong
    /// shape for the transform method (`u(x, 0)`, `u_t(x, 0)`, `u(0, t)`).
    TransformData(String),
    /// `E-PDE-162` — `dsolve` did not solve the transformed equation, or the transformed
    /// equation is not an ODE with constant coefficients in the remaining
    /// variable.
    TransformOde(String),
    /// `E-PDE-163` — the candidate built by the transform method was refuted
    /// by substitution (a kernel that does not satisfy the PDE, a closed form
    /// that misses a condition).
    TransformVerificationFailed(String),
    /// `E-PDE-250` — no Green's function / fundamental solution is implemented
    /// for this operator, dimension or problem (a radial form of a 1-D-only
    /// operator, a symbolic dimension for an operator other than Laplace and
    /// heat, an operator that the requested image construction does not
    /// commute with).
    GreenUnsupported(String),
    /// `E-PDE-251` — malformed input to a Green's-function routine: the wrong
    /// number of space variables for the dimension, a variable that is not a
    /// symbol, a source point of the wrong length, a missing time variable.
    GreenInvalidInput(String),
    /// `E-PDE-252` — the method of images does not apply to this region and
    /// boundary condition (Neumann on a disk, a strip for an operator whose
    /// image series is not implemented, a wall axis out of range).
    GreenRegionUnsupported(String),
    /// `E-PDE-253` — the eigenfunction series cannot be assembled: a squared
    /// norm did not close, eigenvalues below the threshold are present or
    /// conditional, or an eigen-family's own verification did not succeed.
    GreenSeriesUnsupported(String),
    /// `E-PDE-254` — the boundary-value problem has no Green's function: the
    /// homogeneous problem has a nontrivial solution (`λ = 0` is an
    /// eigenvalue; the Wronskian of the two one-sided solutions vanishes).
    /// The modified Green's function of the eigenfunction series applies.
    GreenDoesNotExist(String),
    /// `E-PDE-255` — a constructed Green's function or representation was
    /// refuted by its exact check and withheld.
    GreenVerificationFailed(String),
    /// `E-PDE-256` — `green_solve` does not recognise the problem: not a
    /// linear equation with a tabulated operator, a domain without a Green's
    /// function here, or boundary / initial data of an unsupported shape.
    GreenProblemUnsupported(String),
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
            PdeError::PdsolveUnsupported(d) => ("pdsolve has no method for this problem", d),
            PdeError::NoVerifiedSolution(d) => ("no verified solution", d),
            PdeError::CharacteristicData(d) => ("Cauchy data on a characteristic", d),
            PdeError::CauchyUnsupported(d) => ("unsupported Cauchy data", d),
            PdeError::MalformedIntegral(d) => ("malformed Integral", d),
            PdeError::NotSecondOrderLinear2D(d) => {
                ("not a linear second-order equation in two variables", d)
            }
            PdeError::CharacteristicsNotIntegrated(d) => ("characteristics not integrated", d),
            PdeError::TypeNotUniform(d) => ("type not uniform", d),
            PdeError::LaplaceBudgetExhausted(d) => ("Laplace cascade budget exhausted", d),
            PdeError::NoClosedForm(d) => ("no closed-form solution found", d),
            PdeError::SecondOrderNotVerified(d) => ("candidate not verified", d),
            PdeError::QuadratureOfArbitraryFunction(d) => ("quadrature has no closed form", d),
            PdeError::CauchyDataUnsupported(d) => ("Cauchy data unsupported", d),
            PdeError::NotLaplaceForm(d) => ("not in Laplace form", d),
            PdeError::SlUnsupported(d) => ("Sturm–Liouville problem unsupported", d),
            PdeError::SlInvalidBoundary(d) => ("invalid Sturm–Liouville boundary condition", d),
            PdeError::SlDegenerate(d) => ("degenerate Sturm–Liouville problem", d),
            PdeError::SlVerificationFailed(d) => ("eigenpair failed verification", d),
            PdeError::TransformMethodUnsupported(d) => ("transform method unsupported", d),
            PdeError::TransformData(d) => ("transform method data", d),
            PdeError::TransformOde(d) => ("transformed equation not solved", d),
            PdeError::TransformVerificationFailed(d) => ("transform solution refuted", d),
            PdeError::GreenUnsupported(d) => ("Green's function unsupported", d),
            PdeError::GreenInvalidInput(d) => ("invalid Green's-function input", d),
            PdeError::GreenRegionUnsupported(d) => ("method of images does not apply", d),
            PdeError::GreenSeriesUnsupported(d) => ("eigenfunction series unavailable", d),
            PdeError::GreenDoesNotExist(d) => ("no Green's function exists", d),
            PdeError::GreenVerificationFailed(d) => ("Green's function refuted", d),
            PdeError::GreenProblemUnsupported(d) => ("green_solve does not apply", d),
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
            PdeError::PdsolveUnsupported(_) => "E-PDE-050",
            PdeError::NoVerifiedSolution(_) => "E-PDE-051",
            PdeError::CharacteristicData(_) => "E-PDE-052",
            PdeError::CauchyUnsupported(_) => "E-PDE-053",
            PdeError::MalformedIntegral(_) => "E-PDE-054",
            PdeError::NotSecondOrderLinear2D(_) => "E-PDE-100",
            PdeError::CharacteristicsNotIntegrated(_) => "E-PDE-101",
            PdeError::TypeNotUniform(_) => "E-PDE-102",
            PdeError::LaplaceBudgetExhausted(_) => "E-PDE-103",
            PdeError::NoClosedForm(_) => "E-PDE-104",
            PdeError::SecondOrderNotVerified(_) => "E-PDE-105",
            PdeError::QuadratureOfArbitraryFunction(_) => "E-PDE-106",
            PdeError::CauchyDataUnsupported(_) => "E-PDE-107",
            PdeError::NotLaplaceForm(_) => "E-PDE-108",
            PdeError::SlUnsupported(_) => "E-PDE-150",
            PdeError::SlInvalidBoundary(_) => "E-PDE-151",
            PdeError::SlDegenerate(_) => "E-PDE-152",
            PdeError::SlVerificationFailed(_) => "E-PDE-153",
            PdeError::TransformMethodUnsupported(_) => "E-PDE-160",
            PdeError::TransformData(_) => "E-PDE-161",
            PdeError::TransformOde(_) => "E-PDE-162",
            PdeError::TransformVerificationFailed(_) => "E-PDE-163",
            PdeError::GreenUnsupported(_) => "E-PDE-250",
            PdeError::GreenInvalidInput(_) => "E-PDE-251",
            PdeError::GreenRegionUnsupported(_) => "E-PDE-252",
            PdeError::GreenSeriesUnsupported(_) => "E-PDE-253",
            PdeError::GreenDoesNotExist(_) => "E-PDE-254",
            PdeError::GreenVerificationFailed(_) => "E-PDE-255",
            PdeError::GreenProblemUnsupported(_) => "E-PDE-256",
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
            let n: u32 = code
                .strip_prefix("E-PDE-")
                .and_then(|d| d.parse().ok())
                .unwrap_or_else(|| panic!("{code} is not an E-PDE code"));
            assert!((1..=49).contains(&n), "{code} outside the M0 range");
            assert!(e.remediation().is_some(), "{code} not registered");
        }
    }

    #[test]
    fn m1_codes_are_registered_in_their_range() {
        let all = [
            PdeError::PdsolveUnsupported(String::new()),
            PdeError::NoVerifiedSolution(String::new()),
            PdeError::CharacteristicData(String::new()),
            PdeError::CauchyUnsupported(String::new()),
            PdeError::MalformedIntegral(String::new()),
        ];
        for e in all {
            let code = e.code();
            let n: u32 = code.strip_prefix("E-PDE-").unwrap().parse().unwrap();
            assert!((50..=99).contains(&n), "{code} outside the M1 range");
            assert!(e.remediation().is_some(), "{code} not registered");
        }
    }
}
