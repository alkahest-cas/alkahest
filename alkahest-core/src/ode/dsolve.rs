//! Classical symbolic ODE solver (`dsolve`).
//!
//! Returns closed-form *general* solutions to ordinary differential equations,
//! introducing integration constants `C1, C2, …` as fresh symbols.
//!
//! # Covered classes
//!
//! **First order** (`y' = …` written as `F(x, y, y') = 0`), tried in this
//! order — see the private `first_order` submodule for why the order is what
//! it is:
//! 1. Clairaut `y = x·y' + f(y')` (the only class nonlinear in `y'`)
//! 2. separable `y' = g(x)·h(y)`
//! 3. linear `y' + p(x)·y = q(x)` (integrating-factor)
//! 4. Bernoulli `y' + p(x)·y = q(x)·yⁿ`
//! 5. exact `M dx + N dy = 0` with `∂M/∂y = ∂N/∂x`, including the two standard
//!    integrating-factor rescues when it is only *near*-exact
//! 6. homogeneous of degree zero `y' = G(y/x)` (substitution `v = y/x`)
//! 7. Riccati `y' = q₀(x) + q₁(x)·y + q₂(x)·y²` **with a polynomial particular
//!    solution** found by ansatz (declined otherwise)
//!
//! A first-order answer may be [`SolutionForm::Implicit`]: separable and exact
//! equations frequently have no closed form for `y`.  An explicit `y(x)` is
//! always preferred when any class produces one.  Such an answer is reported by
//! [`dsolve_with`] as a [`DsolveBranch`]; [`dsolve`], whose [`DsolveSolution`]
//! is an explicit `y(x)` by construction, declines and says so rather than
//! passing a relation off as one.
//!
//! **Second order** (`F(x, y, y', y'') = 0`):
//! - constant coefficients `a·y'' + b·y' + c·y = r(x)` (real distinct / repeated
//!   / complex roots), with `a, b, c` **numeric or symbolic** — the damped
//!   oscillator `y'' + 2ζω y' + ω² y = 0` included
//! - Euler–Cauchy `a·x²·y'' + b·x·y' + c·y = r(x)`
//! - general variable coefficients `a₂(x)y'' + a₁(x)y' + a₀(x)y = r(x)`, when a
//!   first homogeneous solution is found by ansatz — the second then follows by
//!   reduction of order (see the private `variation` submodule)
//!
//! **Higher order**: constant-coefficient `Σ aₖ y^(k) = r(x)`, solved through
//! the characteristic polynomial (rational + quadratic factorization;
//! irreducible factors of degree ≥ 3 are declined).  With symbolic
//! coefficients the same polynomial is rooted in closed form up to degree two
//! after a `λᵏ` factor is peeled off, and declined above that.
//!
//! **Systems** `y' = A·y + f(t)` live in [`mod@system`], reached through
//! [`system::dsolve_system`] over the crate's [`ODE`](crate::ode::ODE) type —
//! not through [`dsolve`], whose input names a single unknown.
//!
//! # Parameter branches
//!
//! A symbolic coefficient makes the *multiplicity structure* of the
//! characteristic polynomial parameter dependent, and therefore undecidable:
//! `y'' + 2ζω y' + ω² y = 0` has two distinct roots for `ζ ≠ 1` and a double
//! root at `ζ = 1`.  The returned family satisfies the equation identically
//! either way — that is what the verification gate checks — but it is the
//! *general* solution only on one side.  Which assumption was made is reported,
//! never left implicit: coarsely in [`DsolveSolution::method`], and exactly in
//! [`DsolveReport::side_conditions`] / [`DsolveReport::notes`] from
//! [`dsolve_with`], which also consults caller-stated facts before assuming
//! anything.
//!
//! For every linear class the forcing term `r(x)` is closed either by
//! undetermined coefficients (cheap, exact, but only for
//! polynomial × exp × sin/cos) or, failing that, by variation of parameters at
//! the equation's own order — which is what admits forcings no ansatz can
//! express: `sec x`, `tan x`, `1/(1 + eˣ)`, `log x`, arbitrary rational.
//!
//! # Verification gate
//!
//! *Every* returned solution is verified by substitution: the candidate `y(x)`
//! (and its derivatives) are substituted into the original equation, the
//! residual is simplified, and accepted only when it is the symbolic zero or
//! numerically `≈ 0` at several sample points over random constant values —
//! and, when the equation carries free parameters, over sampled parameter
//! values too, evaluated in ℂ so the complex branch of a symbolic-coefficient
//! answer is reachable.  A candidate that fails verification causes [`dsolve`]
//! to decline (it never returns an unverified solution).
//!
//! An implicit answer `G(x, y) = 0` is verified by the same discipline in the
//! form the implicit function theorem gives: `y' = −Gₓ/G_y` is substituted into
//! the equation and the result must vanish identically in `(x, y)` — which says
//! precisely that *every* level set of `G` solves the ODE.  See `verify`.
//!
//! # Quadratures
//!
//! Closed forms that require an integral defer to the existing
//! [`mod@crate::integrate`] engine.  If a required integral does not close in
//! elementary form, the class is declined (no unevaluated-integral output).
//!
//! `dsolve` manufactures its own integrands — `exp(∫p dx)·q`, `Wₖ·g/W` — and a
//! manufactured integrand is not in normal form: `e^{−log x}` rather than
//! `1/x`, `e^{x}·e^{−x}` rather than `1`, `cos²x + sin²x` rather than `1`.  The
//! integration engine is form-sensitive enough that this decides whether an
//! elementary integral closes, so the private `integrate_or_decline` helper
//! tries each integrand in several equal-valued spellings and takes the first
//! that closes.  Set `ALKAHEST_DSOLVE_TRACE` in a test build to print every
//! integral that no spelling closed.

use crate::deriv::SideCondition;
use crate::diff::diff;
use crate::integrate::engine::integrate;
use crate::kernel::eval_const::try_expr_f64;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::assumptions::AssumptionContext;
use crate::simplify::engine::{distribute_recip, simplify, simplify_expanded};
use std::collections::HashMap;
use std::fmt;

mod constant_coeff;
#[cfg(test)]
mod corpus;
mod first_order;
pub mod system;
mod variation;
mod verify;

#[cfg(test)]
pub(crate) use verify::solution_is_verified;
pub(crate) use verify::{implicit_relation_is_zero, residual_is_zero};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Input description of a scalar ODE for [`dsolve`].
///
/// The equation is supplied as a single expression `equation` that is taken to
/// equal zero, written in terms of the symbols `x` (independent variable), `y`
/// (the unknown `y(x)`), and the derivative symbols in `derivs`
/// (`derivs[0] = y'`, `derivs[1] = y''`, …).  The `order` equals
/// `derivs.len()`.
///
/// Use [`OdeInput::first_order`] / [`OdeInput::second_order`] /
/// [`OdeInput::higher_order`] to build instances; they allocate the derivative
/// symbols with the conventional names `y'`, `y''`, ….
#[derive(Clone, Debug)]
pub struct OdeInput {
    /// Independent variable, e.g. `x`.
    pub x: ExprId,
    /// Dependent variable `y` (representing `y(x)`).
    pub y: ExprId,
    /// Derivative symbols `[y', y'', …]`.
    pub derivs: Vec<ExprId>,
    /// The equation, interpreted as `equation = 0`.
    pub equation: ExprId,
}

impl OdeInput {
    fn deriv_symbol(y: ExprId, k: usize, pool: &ExprPool) -> ExprId {
        let base = pool.with(y, |d| match d {
            ExprData::Symbol { name, .. } => name.clone(),
            _ => "y".to_string(),
        });
        let primes = "'".repeat(k);
        pool.symbol(format!("{base}{primes}"), Domain::Real)
    }

    /// Build a first-order input `equation(x, y, y') = 0`.
    ///
    /// Returns `(input, y')` so the caller can build the equation referring to
    /// the freshly created derivative symbol.
    pub fn first_order(x: ExprId, y: ExprId, pool: &ExprPool) -> (Self, ExprId) {
        let yp = Self::deriv_symbol(y, 1, pool);
        (
            OdeInput {
                x,
                y,
                derivs: vec![yp],
                equation: pool.integer(0_i32),
            },
            yp,
        )
    }

    /// Build a second-order input `equation(x, y, y', y'') = 0`.
    ///
    /// Returns `(input, y', y'')`.
    pub fn second_order(x: ExprId, y: ExprId, pool: &ExprPool) -> (Self, ExprId, ExprId) {
        let yp = Self::deriv_symbol(y, 1, pool);
        let ypp = Self::deriv_symbol(y, 2, pool);
        (
            OdeInput {
                x,
                y,
                derivs: vec![yp, ypp],
                equation: pool.integer(0_i32),
            },
            yp,
            ypp,
        )
    }

    /// Build an `order`-th order input.  Returns `(input, derivs)` where
    /// `derivs[k]` is the `(k+1)`-th derivative symbol.
    pub fn higher_order(
        x: ExprId,
        y: ExprId,
        order: usize,
        pool: &ExprPool,
    ) -> (Self, Vec<ExprId>) {
        assert!(order >= 1, "ODE order must be ≥ 1");
        let derivs: Vec<ExprId> = (1..=order)
            .map(|k| Self::deriv_symbol(y, k, pool))
            .collect();
        (
            OdeInput {
                x,
                y,
                derivs: derivs.clone(),
                equation: pool.integer(0_i32),
            },
            derivs,
        )
    }

    /// Replace the equation expression.
    pub fn with_equation(mut self, equation: ExprId) -> Self {
        self.equation = equation;
        self
    }

    /// ODE order.
    pub fn order(&self) -> usize {
        self.derivs.len()
    }
}

/// How a general solution is written down.
///
/// Separable and exact equations routinely have no closed form for `y`: the
/// answer is a relation `G(x, y) = 0` that the solution curves satisfy, and
/// refusing to return one would refuse most of the two classes.  The two cases
/// are kept in distinct variants rather than in one `ExprId` field so that a
/// caller cannot read a relation as if it were `y(x)` — the single most likely
/// way for this addition to produce a wrong answer downstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionForm {
    /// `y(x) = expr`, with `expr` free of `y`.
    Explicit(ExprId),
    /// `expr(x, y) = 0`, an implicit general solution that could not be solved
    /// for `y`.  Every solution curve of the ODE in the region satisfies it,
    /// and every level set of it solves the ODE — the latter is what the
    /// verification gate checks, by substituting `y' = −Gₓ/G_y`.
    Implicit(ExprId),
}

/// A general solution returned by [`dsolve`]: always an explicit `y(x)`.
///
/// The shape of this struct is frozen: it is exhaustively constructible
/// through the public API, so adding a field to it — one for the implicit form,
/// say — is a major semver break
/// (`cargo-semver-checks::constructible_struct_adds_field`), and so is removing
/// or renaming one.  A branch that may be implicit is a [`DsolveBranch`],
/// reported by [`dsolve_with`].  The separation is not only a versioning
/// artefact: it is also what stops a caller reading a relation `G(x, y) = 0`
/// out of a field named `y_of_x`.
#[derive(Clone, Debug)]
pub struct DsolveSolution {
    /// The solution expression for `y(x)` (the right-hand side of `y(x) = …`),
    /// containing the integration constants in [`Self::constants`].
    pub y_of_x: ExprId,
    /// The fresh constant symbols `C1, C2, …` appearing in [`Self::y_of_x`].
    pub constants: Vec<ExprId>,
    /// Short label of the solving method (e.g. `"separable"`).
    pub method: &'static str,
}

/// A general-solution branch as [`dsolve_with`] reports it: explicit `y(x)` or
/// an implicit relation.
///
/// The two cases are kept in distinct [`SolutionForm`] variants rather than in
/// one `ExprId` field so that a caller cannot read a relation as if it were
/// `y(x)` — the single most likely way for the implicit form to produce a wrong
/// answer downstream.
#[derive(Clone, Debug)]
pub struct DsolveBranch {
    /// Explicit `y(x)` or an implicit relation; see [`SolutionForm`].
    pub form: SolutionForm,
    /// The fresh constant symbols `C1, C2, …` appearing in [`Self::form`].
    pub constants: Vec<ExprId>,
    /// Short label of the solving method (e.g. `"separable"`).
    pub method: &'static str,
}

impl DsolveBranch {
    /// Build an explicit branch `y(x) = y_of_x`.
    pub fn explicit(y_of_x: ExprId, constants: Vec<ExprId>, method: &'static str) -> Self {
        DsolveBranch {
            form: SolutionForm::Explicit(y_of_x),
            constants,
            method,
        }
    }

    /// Build an implicit branch `relation(x, y) = 0`.
    pub fn implicit(relation: ExprId, constants: Vec<ExprId>, method: &'static str) -> Self {
        DsolveBranch {
            form: SolutionForm::Implicit(relation),
            constants,
            method,
        }
    }

    /// This branch as a [`DsolveSolution`], or `None` when it is implicit.
    ///
    /// The `None` is the whole point: an implicit relation has no `y_of_x` to
    /// put in a [`DsolveSolution`], and inventing one is exactly the mistake
    /// the two types exist to prevent.
    pub fn into_solution(self) -> Option<DsolveSolution> {
        match self.form {
            SolutionForm::Explicit(y_of_x) => Some(DsolveSolution {
                y_of_x,
                constants: self.constants,
                method: self.method,
            }),
            SolutionForm::Implicit(_) => None,
        }
    }

    /// The explicit `y(x)`, or `None` when the branch is an implicit relation.
    pub fn y_of_x(&self) -> Option<ExprId> {
        match self.form {
            SolutionForm::Explicit(e) => Some(e),
            SolutionForm::Implicit(_) => None,
        }
    }

    /// The implicit relation `G(x, y)` (read as `G = 0`), or `None` when the
    /// solution is explicit.
    pub fn implicit_relation(&self) -> Option<ExprId> {
        match self.form {
            SolutionForm::Implicit(e) => Some(e),
            SolutionForm::Explicit(_) => None,
        }
    }

    /// Is this an explicit `y(x)`?
    pub fn is_explicit(&self) -> bool {
        matches!(self.form, SolutionForm::Explicit(_))
    }

    /// Human-readable rendering: `y = …` for an explicit solution, `0 = …` for
    /// an implicit relation.  For diagnostics and reports; the two forms are
    /// deliberately not interchangeable as expressions.
    pub fn render(&self, pool: &ExprPool) -> String {
        match self.form {
            SolutionForm::Explicit(e) => format!("y = {}", pool.display(e)),
            SolutionForm::Implicit(e) => format!("0 = {}", pool.display(e)),
        }
    }
}

/// The result of [`dsolve`]: zero or more general-solution branches.
#[derive(Clone, Debug)]
pub struct DsolveResult {
    /// General-solution branches.  Most classes return exactly one branch.
    pub solutions: Vec<DsolveSolution>,
}

/// Errors / declines from [`dsolve`].
///
/// # Why the refined declines are not variants
///
/// This enum is public and exhaustive, so a new variant is a major semver break
/// (`cargo-semver-checks::enum_variant_added`) and marking it
/// `#[non_exhaustive]` now is one too.  The two declines that earn their own
/// error code — a recognised class whose quadrature did not close (`E-ODE-013`)
/// and a recognised class missing the seed its method needs (`E-ODE-014`) —
/// therefore live *inside* [`Self::Unsupported`], tagged by a fixed prefix on
/// the message that [`code`](crate::errors::AlkahestError::code) reads back.
/// Nothing about the diagnostic is lost: the message still names the class and
/// quotes the failing integral or the missing particular solution, `Display`
/// renders it exactly as a dedicated variant would, and
/// [`Self::is_quadrature_failure`] / [`Self::is_missing_particular_solution`]
/// let a caller branch on the distinction.
///
/// Build the tagged forms only through [`Self::quadrature_failed`] and
/// [`Self::no_particular_solution`] so the tag and the code cannot drift apart.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DsolveError {
    /// The ODE did not match any implemented solvable class, or a required
    /// quadrature did not close in elementary form.
    Unsupported(String),
    /// A candidate closed form was produced but failed the substitution
    /// verification gate (so it is withheld rather than returned wrong).
    VerificationFailed(String),
    /// Differentiation of an intermediate expression failed.
    DiffError(String),
}

impl DsolveError {
    /// Message tag for the `E-ODE-013` decline.
    const QUADRATURE_TAG: &'static str = "required quadrature did not close: ";
    /// Message tag for the `E-ODE-014` decline.
    const NO_PARTICULAR_TAG: &'static str = "no particular solution available: ";

    /// The equation *was* recognised as a solvable class, but a quadrature the
    /// method needs did not close in elementary form.  Strictly more
    /// informative than a bare [`Self::Unsupported`]: `detail` names the class
    /// and quotes the integral that failed, so the decline points at the
    /// integration engine rather than at the classifier.
    pub fn quadrature_failed(detail: impl fmt::Display) -> Self {
        DsolveError::Unsupported(format!("{}{detail}", Self::QUADRATURE_TAG))
    }

    /// The equation was recognised as a class whose solution method needs a
    /// seed the solver could not produce — for a Riccati equation, a particular
    /// solution.  No general-Riccati attempt is made; without a particular
    /// solution the closed form is in terms of solutions of an associated
    /// second-order linear ODE (Airy functions for `y' = y² + x`), which this
    /// solver does not emit.
    pub fn no_particular_solution(detail: impl fmt::Display) -> Self {
        DsolveError::Unsupported(format!("{}{detail}", Self::NO_PARTICULAR_TAG))
    }

    /// Was this decline a quadrature that did not close (`E-ODE-013`)?
    pub fn is_quadrature_failure(&self) -> bool {
        matches!(self, DsolveError::Unsupported(m) if m.starts_with(Self::QUADRATURE_TAG))
    }

    /// Was this decline a missing particular solution (`E-ODE-014`)?
    pub fn is_missing_particular_solution(&self) -> bool {
        matches!(self, DsolveError::Unsupported(m) if m.starts_with(Self::NO_PARTICULAR_TAG))
    }
}

impl fmt::Display for DsolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // A tagged message already begins with its own description, so
            // prefixing "unsupported ODE:" would only repeat it.
            DsolveError::Unsupported(m)
                if self.is_quadrature_failure() || self.is_missing_particular_solution() =>
            {
                write!(f, "dsolve: {m}")
            }
            DsolveError::Unsupported(m) => write!(f, "dsolve: unsupported ODE: {m}"),
            DsolveError::VerificationFailed(m) => {
                write!(f, "dsolve: candidate failed verification: {m}")
            }
            DsolveError::DiffError(m) => write!(f, "dsolve: differentiation error: {m}"),
        }
    }
}

impl std::error::Error for DsolveError {}

impl crate::errors::AlkahestError for DsolveError {
    fn code(&self) -> &'static str {
        match self {
            _ if self.is_quadrature_failure() => "E-ODE-013",
            _ if self.is_missing_particular_solution() => "E-ODE-014",
            DsolveError::Unsupported(_) => "E-ODE-010",
            DsolveError::VerificationFailed(_) => "E-ODE-011",
            DsolveError::DiffError(_) => "E-ODE-012",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            _ if self.is_quadrature_failure() => Some(
                "the ODE was classified, but the integral the method needs is not \
                 elementary for this integrator; the message names the class and the \
                 integrand that failed",
            ),
            _ if self.is_missing_particular_solution() => Some(
                "supply a particular solution, or expect a closed form outside the \
                 elementary/special-function vocabulary this solver emits",
            ),
            DsolveError::Unsupported(_) => Some(
                "the ODE is outside the implemented classical classes, or a required \
                 integral is non-elementary; check the equation form",
            ),
            DsolveError::VerificationFailed(_) => Some(
                "the solver found a candidate that did not verify by substitution; \
                 this is reported rather than returned as a (possibly wrong) answer",
            ),
            DsolveError::DiffError(_) => {
                Some("ensure the equation only contains differentiable functions")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Solve a scalar ODE in closed form, returning the general solution(s).
///
/// Dispatches on the ODE order and structure to the implemented classical
/// methods.  Every returned solution is verified by substitution (see the
/// [module docs](self)); unverifiable candidates are withheld and the relevant
/// class declines.
///
/// # Errors
///
/// Returns [`DsolveError::Unsupported`] when the equation is outside the
/// implemented classes or a required quadrature is non-elementary, and
/// [`DsolveError::VerificationFailed`] when a candidate could not be verified.
///
/// A [`DsolveSolution`] is always an explicit `y(x)`.  When every verified
/// branch is an implicit relation `G(x, y) = 0` — routine for separable and
/// exact equations — this declines rather than returning an empty
/// [`DsolveResult`], and the message points at [`dsolve_with`], which reports
/// the relation.  An `Ok` therefore always carries at least one branch.
pub fn dsolve(input: &OdeInput, pool: &ExprPool) -> Result<DsolveResult, DsolveError> {
    let report = dsolve_with(input, &AssumptionContext::new(), pool)?;
    let implicit: Vec<String> = report
        .branches
        .iter()
        .filter(|b| !b.is_explicit())
        .map(|b| b.render(pool))
        .collect();
    let solutions: Vec<DsolveSolution> = report
        .branches
        .into_iter()
        .filter_map(DsolveBranch::into_solution)
        .collect();
    if solutions.is_empty() {
        return Err(DsolveError::Unsupported(format!(
            "the general solution is an implicit relation, not an explicit y(x): {}; \
             use dsolve_with to obtain it",
            implicit.join("; ")
        )));
    }
    Ok(DsolveResult { solutions })
}

/// [`dsolve`], plus the conditions under which the returned branches are the
/// **general** solution and any facts the caller has stated.
///
/// # Why this exists rather than a field on [`DsolveSolution`]
///
/// A constant-coefficient equation with *symbolic* coefficients has a
/// characteristic polynomial whose multiplicity structure is parameter
/// dependent: `y'' + 2ζω y' + ω² y = 0` has two distinct roots for `ζ ≠ 1` and
/// one double root at `ζ = 1`, and no amount of simplification decides which,
/// because both happen.  The two-exponential form
/// `C₁e^{r₊x} + C₂e^{r₋x}` is *a* solution for every parameter value — it
/// satisfies the equation identically — but it is the **general** solution only
/// where the roots are distinct; at `ζ = 1` the two branches coincide and the
/// missing second solution is `x·e^{rx}`.
///
/// Returning that form with no way to say so would be exactly the silent
/// assumption this library refuses to make, so the assumption travels with the
/// answer: [`DsolveReport::side_conditions`] carries `discriminant ≠ 0` as a
/// [`SideCondition`], and [`DsolveReport::notes`] spells out what happens where
/// it fails.  Callers of plain [`dsolve`] still see it, coarsely, in
/// [`DsolveSolution::method`], which names the branch
/// (`"constant_coefficient_symbolic"` vs `"…_repeated_root"`).
///
/// `assumptions` is consulted before the branch is guessed: a stated
/// `discriminant > 0` or `discriminant ≠ 0` removes the side condition, and a
/// stated `−discriminant > 0` selects the real `e^{αx}(C₁cos βx + C₂sin βx)`
/// form instead of the complex exponentials.
///
/// # Errors
///
/// As [`dsolve`].
pub fn dsolve_with(
    input: &OdeInput,
    assumptions: &AssumptionContext,
    pool: &ExprPool,
) -> Result<DsolveReport, DsolveError> {
    let mut gen = ConstGen::new(input, pool);
    let mut ctx = SolveCtx {
        assumptions,
        conds: Conditions::default(),
    };
    let branches = match input.order() {
        1 => first_order::solve(input, &mut gen, pool),
        2 => constant_coeff::solve_second_order(input, &mut gen, &mut ctx, pool),
        n if n >= 3 => constant_coeff::solve_higher_order(input, n, &mut gen, &mut ctx, pool),
        _ => Err(DsolveError::Unsupported("order 0 ODE".to_string())),
    }?;
    Ok(DsolveReport {
        branches,
        side_conditions: ctx.conds.side,
        notes: ctx.conds.notes,
    })
}

/// The result of [`dsolve_with`].
#[derive(Clone, Debug)]
pub struct DsolveReport {
    /// The verified general-solution branches, explicit or implicit.  Never
    /// empty on `Ok`: a class that produced nothing declines instead.
    pub branches: Vec<DsolveBranch>,
    /// Conditions under which [`Self::branches`] is the *general* solution.
    ///
    /// Empty means unconditional.  A non-empty list is not a hedge about
    /// correctness — every branch here has been verified by substitution — it
    /// is about *completeness*: where a condition fails, the returned family
    /// still solves the equation but no longer spans every solution.
    pub side_conditions: Vec<SideCondition>,
    /// Prose for the cases [`Self::side_conditions`] excludes.
    pub notes: Vec<String>,
}

/// Per-call state threaded through the solving classes: what the caller has
/// asserted, and what the class had to assume.
pub(crate) struct SolveCtx<'a> {
    pub(crate) assumptions: &'a AssumptionContext,
    pub(crate) conds: Conditions,
}

/// Conditions a class attached to its answer.
#[derive(Default, Debug, Clone)]
pub(crate) struct Conditions {
    pub(crate) side: Vec<SideCondition>,
    pub(crate) notes: Vec<String>,
}

impl Conditions {
    pub(crate) fn require_nonzero(&mut self, e: ExprId) {
        let c = SideCondition::NonZero(e);
        if !self.side.contains(&c) {
            self.side.push(c);
        }
    }

    pub(crate) fn note(&mut self, s: String) {
        if !self.notes.contains(&s) {
            self.notes.push(s);
        }
    }
}

/// What the caller's assumptions establish about the sign of a quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AssumedSign {
    /// Strictly positive.
    Positive,
    /// Strictly negative.
    Negative,
    /// Non-zero, sign not determined.
    NonZero,
    /// Nothing follows.
    Unknown,
}

impl SolveCtx<'_> {
    /// What the caller's facts say about the sign of `e`.
    ///
    /// A fact about `u` settles `e` when `e = q·u` for a non-zero rational `q`
    /// small enough to be found by [`constant_ratio`].  That is a deliberately
    /// narrow rule: it covers the spellings the same quantity arrives in
    /// (`ka − ke` against `ke − ka`, a discriminant against a quarter of it),
    /// which is what a caller actually trips over, and it needs no reasoning
    /// beyond one provable identity.
    ///
    /// Deliberately **not** covered: deriving `4ω²(ζ² − 1) < 0` from
    /// `ω > 0 ∧ ζ < 1`.  That is a real sign-decision problem;
    /// `logic::satisfiable` is interval propagation over unbounded boxes and
    /// does not refute the conjunction, and a heuristic that "usually" gets it
    /// right is exactly how a wrong branch is picked silently.  A caller who
    /// needs that branch states the fact about the discriminant itself.
    pub(crate) fn assumed_sign(&self, e: ExprId, pool: &ExprPool) -> AssumedSign {
        let target = expand_powers(e, pool);
        let mut verdict = AssumedSign::Unknown;
        for fact in self.assumptions.facts() {
            let (u, strict_sign) = match fact {
                SideCondition::Positive(id) => (*id, true),
                SideCondition::NonZero(id) => (*id, false),
                SideCondition::InDomain(..) => continue,
            };
            let Some(ratio) = constant_ratio(target, expand_powers(u, pool), pool) else {
                continue;
            };
            if !strict_sign {
                if verdict == AssumedSign::Unknown {
                    verdict = AssumedSign::NonZero;
                }
                continue;
            }
            return if ratio > 0.0 {
                AssumedSign::Positive
            } else {
                AssumedSign::Negative
            };
        }
        verdict
    }

    /// Has the caller ruled out `e = 0`?
    pub(crate) fn asserts_nonzero(&self, e: ExprId, pool: &ExprPool) -> bool {
        !matches!(self.assumed_sign(e, pool), AssumedSign::Unknown)
    }
}

/// Is `e ≠ 0` already established, without any caller assumption?
///
/// True for a non-zero literal, and for any **closed** expression (no free
/// symbols) the zero test can certify non-zero — `−2√(−1)`, the eigenvalue gap
/// of a rotation.  For an expression that *does* mention a free symbol,
/// `ZeroStatus::NonZero` means only "not identically zero as a function", which
/// is not the same claim and must not be read as one: `ζ² − 1` is not
/// identically zero and still vanishes at `ζ = 1`.
pub(crate) fn is_settled_nonzero(e: ExprId, pool: &ExprPool) -> bool {
    crate::matrix::zero_test::settled_nonzero(e, pool)
}

/// `q` such that `a = q·b`, for a small non-zero rational `q`, or `None`.
///
/// Asking `simplify(a/b)` instead does not work: `(ke − ka)·(ka − ke)⁻¹` is a
/// `Mul` of an `Add` and a `Pow` of an `Add`, and the default rule set has no
/// reason to cancel it.  Testing `a − q·b = 0` for a fixed candidate list turns
/// the question into one the zero test *can* answer, and the answer it gives is
/// a proof rather than a numeric coincidence.
fn constant_ratio(a: ExprId, b: ExprId, pool: &ExprPool) -> Option<f64> {
    if let Some(v) = try_expr_f64(div(a, b, pool), pool) {
        if v != 0.0 && v.is_finite() {
            return Some(v);
        }
    }
    const NUMS: [i64; 8] = [1, -1, 2, -2, 3, -3, 4, -4];
    const DENS: [i64; 4] = [1, 2, 3, 4];
    for num in NUMS {
        for den in DENS {
            let q = if den == 1 {
                pool.integer(num)
            } else {
                pool.rational(num, den)
            };
            let diff = expand_powers(
                pool.add(vec![a, pool.mul(vec![pool.integer(-1_i32), q, b])]),
                pool,
            );
            if crate::matrix::zero_test::zero_status(pool, diff).is_proven_zero() {
                return Some(num as f64 / den as f64);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Fresh-constant generator (collision-free with user symbols)
// ---------------------------------------------------------------------------

/// Allocates fresh integration-constant symbols `C1, C2, …`, skipping any name
/// already present in the input equation so user symbols never collide.
pub(crate) struct ConstGen {
    next: usize,
    used: std::collections::HashSet<String>,
    /// Names this generator handed out, in order — the only ones a rollback may
    /// take back (a user symbol that happened to be called `C2` is in `used`
    /// too, and releasing it would let a later class collide with it).
    issued: Vec<String>,
}

impl ConstGen {
    fn new(input: &OdeInput, pool: &ExprPool) -> Self {
        let mut used = std::collections::HashSet::new();
        collect_symbol_names(input.equation, pool, &mut used);
        ConstGen {
            next: 1,
            used,
            issued: Vec::new(),
        }
    }

    /// A generator that avoids an explicitly supplied set of names.
    ///
    /// [`Self::new`] derives the set from an [`OdeInput`]; a system has no
    /// single equation to walk, so its caller collects the names itself.
    pub(crate) fn with_used(used: std::collections::HashSet<String>) -> Self {
        ConstGen {
            next: 1,
            used,
            issued: Vec::new(),
        }
    }

    /// Return a fresh constant symbol whose name (`C{n}`) is not already used.
    pub(crate) fn fresh(&mut self, pool: &ExprPool) -> ExprId {
        loop {
            let name = format!("C{}", self.next);
            self.next += 1;
            if !self.used.contains(&name) {
                self.used.insert(name.clone());
                self.issued.push(name.clone());
                return pool.symbol(name, Domain::Real);
            }
        }
    }

    /// Mark for a later [`Self::rollback`].
    pub(crate) fn checkpoint(&self) -> ConstMark {
        ConstMark {
            next: self.next,
            issued: self.issued.len(),
        }
    }

    /// Release the constants allocated since `mark`.
    ///
    /// A classification cascade allocates a constant for a class that then
    /// fails to close, and without this the answer that a later class finds is
    /// labelled `C2` (or `C4`) — the count of classes that were tried, leaking
    /// the solver's search into the answer and making the output depend on
    /// changes elsewhere in the cascade.  Only names this generator issued are
    /// released.
    pub(crate) fn rollback(&mut self, mark: ConstMark, _pool: &ExprPool) {
        for name in self.issued.drain(mark.issued..) {
            self.used.remove(&name);
        }
        self.next = mark.next;
    }
}

/// Opaque position in a [`ConstGen`]'s allocation sequence.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstMark {
    next: usize,
    issued: usize,
}

fn collect_symbol_names(
    expr: ExprId,
    pool: &ExprPool,
    out: &mut std::collections::HashSet<String>,
) {
    pool.with(expr, |d| match d {
        ExprData::Symbol { name, .. } => {
            out.insert(name.clone());
        }
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            for &a in args {
                collect_symbol_names(a, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect_symbol_names(*base, pool, out);
            collect_symbol_names(*exp, pool, out);
        }
        _ => {}
    });
}

// ---------------------------------------------------------------------------
// Shared small helpers (used across submodules)
// ---------------------------------------------------------------------------

/// Simplify with distribution (expanded normal form).  The classification
/// logic relies on polynomial-in-`x`/`y` terms being flattened (e.g.
/// `−1·(−3y−x)` becoming `3y + x`) so coefficient extraction by structural
/// inspection works.
pub(crate) fn simp(expr: ExprId, pool: &ExprPool) -> ExprId {
    simplify_expanded(expr, pool).value
}

/// Plain (non-expanding) simplify, for the final residual zero-check where
/// expansion is not required.
pub(crate) fn simp_plain(expr: ExprId, pool: &ExprPool) -> ExprId {
    simplify(expr, pool).value
}

/// `diff(expr, var).value`, mapping `DiffError` into `DsolveError`.
pub(crate) fn ddx(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, DsolveError> {
    diff(expr, var, pool)
        .map(|d| d.value)
        .map_err(|e| DsolveError::DiffError(e.to_string()))
}

/// Integrate `expr` in `var`; map any decline to `Unsupported` so the caller
/// declines the whole class (we never emit unevaluated-integral output).
///
/// The integrand is tried in several *spellings* (see [`integrand_spellings`]):
/// `dsolve` manufactures its integrands — `exp(∫p dx)·q`, `y₂·g/W` — and they
/// arrive carrying artefacts (`e^{x}·e^{−x}`, `cos²x + sin²x`) that the default
/// rule set does not cancel.  The integration engine sees an integrand it
/// cannot close where the *same function*, spelled normally, is trivial.  Each
/// spelling is mathematically equal to the original on the solution domain, so
/// the first one that closes is the answer.
pub(crate) fn integrate_or_decline(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    integrate_first_of(&[expr], var, pool)
}

/// `integrate_or_decline` over several *constructions* of the same integrand.
///
/// A caller that can build the integrand more than one way (variation of
/// parameters can divide by the raw Wronskian or by the normalised one) passes
/// all of them; each is expanded into its [`integrand_spellings`] and the
/// first that closes wins.  All candidates must be equal as functions.
pub(crate) fn integrate_first_of(
    exprs: &[ExprId],
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, DsolveError> {
    let mut last: Option<String> = None;
    let mut tried: Vec<ExprId> = Vec::new();
    // Stage 1: every candidate exactly as the caller built it.  Rewriting is
    // only worth its cost once the engine has actually refused, and the common
    // case is that it does not.
    for &expr in exprs {
        if tried.contains(&expr) {
            continue;
        }
        tried.push(expr);
        match integrate(expr, var, pool) {
            Ok(d) => return Ok(simp(d.value, pool)),
            Err(e) => last = last.or(Some(e.to_string())),
        }
    }
    // Stage 2: the rewritings.
    for &expr in exprs {
        for cand in integrand_spellings(expr, pool) {
            if tried.contains(&cand) {
                continue;
            }
            tried.push(cand);
            match integrate(cand, var, pool) {
                Ok(d) => return Ok(simp(d.value, pool)),
                Err(e) => last = last.or(Some(e.to_string())),
            }
        }
    }
    #[cfg(test)]
    if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
        // Every line here is an integral `dsolve` needs and `integrate` does not
        // close — the actionable feedback list for the integration engine.
        for cand in tried {
            eprintln!(
                "INT_DECLINE\td/d{}\t{}",
                pool.display(var),
                pool.display(cand)
            );
        }
    }
    Err(DsolveError::Unsupported(format!(
        "required integral did not close: {}",
        last.unwrap_or_else(|| "no candidate form".to_string())
    )))
}

/// The most-normalised spelling of `expr` (the last [`integrand_spellings`]
/// candidate).  Used for expressions that are *not* about to be integrated —
/// the coefficients `P` and `Q` of a normalised second-order equation, say —
/// where there is no engine to fall through on the caller's behalf.
pub(crate) fn normalized(expr: ExprId, pool: &ExprPool) -> ExprId {
    let cands = integrand_spellings(expr, pool);
    *cands.last().unwrap_or(&expr)
}

/// Equal-valued rewritings of an integrand, cheapest first.
///
/// 1. the expression as given;
/// 2. with reciprocals distributed over products ([`distribute_recip`]);
/// 3. after the log/exp rule set (`e^{a}·e^{b} → e^{a+b}`, `log(e^u) → u`),
///    which collapses the integrating-factor artefacts;
/// 4. additionally after the trig normal form (`cos²u + sin²u → 1`), which
///    collapses a Wronskian of `{cos, sin}` — only attempted when the
///    expression actually mentions sin/cos, since that pass expands products.
///
/// **The list is ordered, not ranked, and the original comes first on
/// purpose.** Normalising is not monotone for the integration engine:
/// `∫ sin x·tan x/(cos²x + sin²x) dx` closes and `∫ sin x·tan x dx` — the same
/// integrand with the redundant `1` cancelled — does not. Until that is fixed
/// upstream, dropping the un-normalised form would lose ODEs that currently
/// solve, so every spelling is tried.
///
/// Duplicates are dropped, so a fully-normalised expression costs one call.
fn integrand_spellings(expr: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = vec![expr];
    let push = |e: ExprId, out: &mut Vec<ExprId>| {
        if !out.contains(&e) {
            out.push(e);
        }
    };
    let dr = simp(distribute_recip(expr, pool), pool);
    push(dr, &mut out);
    for base in [expr, dr] {
        let le = simp(
            crate::simplify::engine::simplify_log_exp(base, pool, &[]).value,
            pool,
        );
        push(le, &mut out);
        if mentions_sin_cos(le, pool) {
            let tn = simp(
                crate::simplify::engine::simplify_trig_normal_form(le, pool).value,
                pool,
            );
            push(tn, &mut out);
        }
    }
    out
}

/// `simp`, with integer powers of products distributed first.
///
/// `simplify_expanded` does **not** turn `(z·ω)²` into `z²ω²` — the `Pow` wraps
/// a whole `Mul`, so the exponent never reaches the factors — and a
/// characteristic discriminant built from `(2ζω)² − 4ω²` therefore keeps a
/// `(ζ·ω)²` that no later step cancels against a `ζ²ω²` written by the caller.
/// Every quantity whose *vanishing* is going to be tested — a discriminant, an
/// eigenvalue gap — goes through here first, so the zero test and the
/// assumption matcher see one spelling rather than two.
pub(crate) fn expand_powers(expr: ExprId, pool: &ExprPool) -> ExprId {
    crate::simplify::engine::expand_powers(expr, pool)
}

fn mentions_sin_cos(expr: ExprId, pool: &ExprPool) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Func { name, args } => {
            name == "sin" || name == "cos" || args.iter().any(|&a| mentions_sin_cos(a, pool))
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| mentions_sin_cos(a, pool))
        }
        ExprData::Pow { base, exp } => {
            mentions_sin_cos(*base, pool) || mentions_sin_cos(*exp, pool)
        }
        _ => false,
    })
}

/// Build `exp(arg)`, folding logarithmic summands into powers:
/// `exp(c·log(u) + rest) → u^c · exp(rest)`.
///
/// Every integrating factor `μ = exp(∫p dx)` in this module goes through here.
/// The default rule set will not apply `exp(log u) → u` (it is a branch-cut
/// identity, sound only for `u > 0`), so without this fold the linear class
/// manufactures `μ = e^{−log x}` and then asks the integration engine for
/// `∫ q·e^{−log x} dx` where it means `∫ q/x dx` — and gets a decline for an
/// integral that is elementary.
///
/// Taking `u > 0` is the same convention that writing `log(u)` in the
/// antiderivative already commits to, and it is not load-bearing for
/// correctness: every candidate solution still has to pass
/// [`residual_is_zero`], which samples the residual at positive `x`.
pub(crate) fn exp_of(arg: ExprId, pool: &ExprPool) -> ExprId {
    let terms: Vec<ExprId> = match pool.get(simp(arg, pool)) {
        ExprData::Add(args) => args,
        _ => vec![simp(arg, pool)],
    };
    let mut factors: Vec<ExprId> = Vec::new();
    let mut rest: Vec<ExprId> = Vec::new();
    for t in terms {
        match log_summand(t, pool) {
            Some((u, c)) => factors.push(pool.pow(u, c)),
            None => rest.push(t),
        }
    }
    if factors.is_empty() {
        return simp(pool.func("exp", vec![arg]), pool);
    }
    if !rest.is_empty() {
        let r = pool.add(rest);
        factors.push(pool.func("exp", vec![r]));
    }
    // A one-element `Mul` is *not* collapsed by `simplify`, and a surviving
    // `Mul([x])` blocks the power collection that would cancel `x·x⁻¹` later.
    if factors.len() == 1 {
        return simp(factors[0], pool);
    }
    simp(pool.mul(factors), pool)
}

/// Recognise `c·log(u)` (or bare `log(u)`, `c = 1`) and return `(u, c)` with
/// `c` a numeric constant expression.
fn log_summand(term: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if let ExprData::Func { name, args } = pool.get(term) {
        if name == "log" && args.len() == 1 {
            return Some((args[0], pool.integer(1_i32)));
        }
    }
    let ExprData::Mul(args) = pool.get(term) else {
        return None;
    };
    let mut inner = None;
    let mut coeff: Vec<ExprId> = Vec::new();
    for a in args {
        match pool.get(a) {
            ExprData::Func { name, args: fargs } if name == "log" && fargs.len() == 1 => {
                if inner.is_some() {
                    return None; // product of two logs — not of this form
                }
                inner = Some(fargs[0]);
            }
            ExprData::Integer(_) | ExprData::Rational(_) => coeff.push(a),
            _ => return None, // a non-constant cofactor: exp(x·log u) is not u^c
        }
    }
    let u = inner?;
    // `simp` leaves a one-element `Mul` standing, and `Mul([−1])` as an
    // exponent is a *different* node from `−1`, which silently defeats the
    // power collection in `x·x⁻¹`.  Collapse it here.
    let c = match coeff.len() {
        0 => pool.integer(1_i32),
        1 => simp(coeff[0], pool),
        _ => simp(pool.mul(coeff), pool),
    };
    Some((u, c))
}

/// Does `expr` contain `needle` as a sub-expression?
pub(crate) fn contains(expr: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if expr == needle {
        return true;
    }
    pool.with(expr, |d| match d {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => {
            args.iter().any(|&a| contains(a, needle, pool))
        }
        ExprData::Pow { base, exp } => {
            contains(*base, needle, pool) || contains(*exp, needle, pool)
        }
        _ => false,
    })
}

/// `a - b`, simplified.
pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let neg_b = pool.mul(vec![pool.integer(-1_i32), b]);
    simp(pool.add(vec![a, neg_b]), pool)
}

/// `a / b`, simplified.
pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let inv_b = pool.pow(b, pool.integer(-1_i32));
    simp(pool.mul(vec![a, inv_b]), pool)
}

/// Substitute a single symbol → replacement, simplifying the result.
pub(crate) fn subs1(expr: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    simp(crate::kernel::subs::subs(expr, &m, pool), pool)
}

/// Is `expr` the literal zero after simplification?
pub(crate) fn is_zero(expr: ExprId, pool: &ExprPool) -> bool {
    let s = simp(expr, pool);
    matches!(pool.get(s), ExprData::Integer(n) if n.0 == 0)
        || matches!(try_expr_f64(s, pool), Some(v) if v == 0.0)
}

#[cfg(test)]
mod tests;
