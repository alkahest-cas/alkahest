//! Probability distributions as symbolic objects, and expectations over them.
//!
//! A [`Distribution`] is a *name plus symbolic parameters*. It carries its own
//! density, its support, and the parameter constraints that make it a
//! distribution at all (`σ > 0`, `a < b`, `0 ≤ p ≤ 1`). Everything derived from
//! it — [`Distribution::mean`], [`Distribution::variance`],
//! [`Distribution::moment`], [`Distribution::cdf`], [`Distribution::quantile`],
//! and the general [`expectation`] — is **checked against a numerical
//! quadrature of its own defining integral before it is returned**. A closed
//! form the checker cannot confirm is withheld ([`ProbError::Unverified`],
//! `E-PROB-005`), never returned with a caveat.
//!
//! # Why a separate module rather than a wider `integrate`
//!
//! An expectation is not an integral the user wrote down; it is an integral
//! this module *constructs* from a distribution, and the construction is where
//! the mistakes live — the wrong Jacobian, the wrong support, the wrong
//! normalising constant, a parameter outside its constraint. Those are exactly
//! the mistakes a symbolic integrator cannot catch, because by the time it sees
//! the integrand the error is already baked in and it will integrate the wrong
//! thing perfectly. So the construction is checked *end to end*, against the
//! definition, numerically, every time.
//!
//! # How an expectation is computed
//!
//! `E[f(X)] = ∫_S f(x) p(x) dx` is real but rarely tractable in `x`. Each
//! continuous distribution therefore declares a **reduction**: an increasing
//! change of variables `x = ξ(z)` under which the density becomes a standard
//! kernel the existing integrator has a chance with
//! (`e^{-z²/2}` for `Normal`/`LogNormal`, `e^{-u}` for `Exponential`/`Gamma`,
//! the identity for `Uniform`/`Beta`). The integrand is pushed through the
//! reduction, expanded into a sum, each term's `z`-free factor is pulled out,
//! and what is left is handed to [`crate::integrate::integrate_definite`].
//!
//! That pipeline can go wrong in four places, so it is **verified against the
//! untransformed definition**: the numeric gate quadratures `f(x) p(x)` over
//! the original support in `x`, not over `z`. A wrong Jacobian, a wrong
//! inverse, a dropped constant or a mis-mapped bound all show up as a
//! disagreement and the answer is refused. See [`verify`].
//!
//! # What refuses, and why that is right
//!
//! | input | refusal |
//! |---|---|
//! | `Normal(μ, 0)`, `Uniform(1, 0)`, `Beta(-1, 2)` | `E-PROB-001`: the parameter is a *number* and it violates the constraint. There is no distribution to ask about |
//! | `Gamma(k, θ).cdf(x)` for symbolic `k` | `E-PROB-004`: the lower incomplete gamma function is not in this library's primitive set, so there is no closed form to return. Integer `k` (Erlang) does close and is returned |
//! | `Normal(μ, σ).quantile(p)` | `E-PROB-004`: needs `erf⁻¹`, which is not a registered primitive |
//! | `E[e^{X²}]` under `Normal(0, 1)` | `E-PROB-006`: the reduction integral `∫ e^{z²} e^{-z²/2} dz` **diverges**. The gate sees the quadrature fail to converge and says so rather than returning the integrator's formal answer |
//! | an integrand the integrator declines | `E-PROB-003`, naming the exact integral that did not close. Not an unevaluated `Integral` object dressed as an answer |
//! | a closed form the quadrature contradicts | `E-PROB-005`. This is the prime-directive case: something *was* computed and is being withheld |
//! | `Uniform(a, b).quantile(2)` | `E-PROB-001`: `p` is a probability. The closed form returns `a + 2(b-a)` there — a point outside the support, offered as a quantile |
//!
//! # What this module deliberately does not do
//!
//! There is no joint distribution, no conditioning, and no dependence
//! structure. [`expectation_affine`] applies linearity — `E[aX + bY] = aE[X] +
//! bE[Y]`, which needs no independence at all — and
//! [`variance_affine_independent`] applies the variance rule under an
//! independence assumption the caller states by choosing that function's name.
//! Anything beyond those two would require a joint-law design, and guessing at
//! one produces wrong covariances silently.

use std::fmt;

use crate::errors::AlkahestError;
use crate::kernel::{Domain, ExprId, ExprPool};

mod charfun;
mod cplx;
mod dists;
mod expect;
mod moments;
mod quad;
#[cfg(test)]
mod tests;
mod verify;

pub use charfun::characteristic_function;
pub use expect::{expectation, expectation_affine, variance_affine_independent};
pub use verify::Evidence;

pub(crate) use quad::numeric_ball;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a probability query did not return a value.
///
/// `#[non_exhaustive]` from birth: a future refusal reason has to be an
/// additive change rather than a semver break.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProbError {
    /// A parameter evaluates to a number that violates the distribution's own
    /// constraint — `σ ≤ 0`, `a ≥ b`, `p ∉ [0, 1]`, a non-integer `n` for
    /// `Binomial`. `E-PROB-001`.
    ///
    /// A *symbolic* parameter is never reported here: it cannot be decided, so
    /// it is carried as a side condition on
    /// [`Distribution::constraints`] instead.
    InvalidParameter {
        /// Which parameter, by name (`"sigma"`, `"a"`, `"p"`, …).
        parameter: &'static str,
        /// The constraint that failed, spelled out.
        requirement: &'static str,
    },
    /// The query is outside this module's supported shape — a general `f` over
    /// a distribution with infinite discrete support, a non-affine argument to
    /// [`expectation_affine`], a moment order this table does not carry.
    /// `E-PROB-002`.
    Unsupported(String),
    /// The reduction integral was built and [`crate::integrate::integrate`]
    /// declined it. The message names the integral. `E-PROB-003`.
    ///
    /// Distinct from [`ProbError::NoClosedForm`] on purpose: this one says
    /// *this* integrator could not do it, which may change; that one says the
    /// closed form needs a function this library does not have.
    IntegralDidNotClose(String),
    /// The quantity has no closed form within the primitive set — the `Gamma`
    /// CDF needs `γ(k, x)`, the `Beta` CDF needs `I_x(α, β)`, the `Normal`
    /// quantile needs `erf⁻¹`. `E-PROB-004`.
    NoClosedForm {
        /// The quantity asked for.
        quantity: &'static str,
        /// The function that would be needed to express it.
        missing: &'static str,
    },
    /// A closed form **was** computed and then **withheld**, because the
    /// numeric gate could not confirm it. `E-PROB-005`.
    ///
    /// Never downgraded to a warning: a caller cannot tell a checked value
    /// from an unchecked one once it is in their hands.
    Unverified(UnverifiedReason),
    /// The defining integral (or series) does not converge, so the quantity
    /// does not exist. `E-PROB-006`.
    ///
    /// The formal symbolic answer for a divergent expectation is typically a
    /// clean, plausible, wrong number — which is the whole reason the gate
    /// runs before the answer is returned rather than after.
    Divergent(String),
}

/// Why the numeric gate declined to confirm a computed closed form.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnverifiedReason {
    /// The closed form and the quadrature of the defining integral disagree by
    /// more than the quadrature's own error estimate allows. The strings are
    /// the two values and the parameter point at which they parted.
    Disagreement(String),
    /// The closed form could not be evaluated to a number at any admissible
    /// parameter point — an unregistered head, or a branch that is not real
    /// there. An unevaluable "answer" reports success and then turns into
    /// `NaN`, which is the failure mode this library exists to avoid.
    ClaimNotEvaluable,
    /// The quadrature did not converge to the precision the comparison needs,
    /// so it cannot *contradict* the closed form either. Silence is not
    /// evidence: the form is withheld rather than passed on a check that could
    /// not fail.
    QuadratureInconclusive,
    /// No admissible parameter point could be constructed — the constraints
    /// involve symbols this module cannot sample.
    NoAdmissiblePoint,
}

impl fmt::Display for ProbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProbError::InvalidParameter {
                parameter,
                requirement,
            } => write!(
                f,
                "prob: parameter `{parameter}` violates the constraint {requirement}"
            ),
            ProbError::Unsupported(msg) => write!(f, "prob: unsupported: {msg}"),
            ProbError::IntegralDidNotClose(msg) => {
                write!(f, "prob: the reduction integral did not close: {msg}")
            }
            ProbError::NoClosedForm { quantity, missing } => write!(
                f,
                "prob: no closed form for {quantity} — it requires {missing}, \
                 which is not a registered primitive"
            ),
            ProbError::Unverified(reason) => match reason {
                UnverifiedReason::Disagreement(detail) => write!(
                    f,
                    "prob: a closed form was computed and withheld — it disagrees \
                     with quadrature of its own defining integral ({detail})"
                ),
                UnverifiedReason::ClaimNotEvaluable => write!(
                    f,
                    "prob: a closed form was computed and withheld — it could not be \
                     evaluated to a number at any admissible parameter point"
                ),
                UnverifiedReason::QuadratureInconclusive => write!(
                    f,
                    "prob: a closed form was computed and withheld — the verifying \
                     quadrature did not converge, so the check could not have failed"
                ),
                UnverifiedReason::NoAdmissiblePoint => write!(
                    f,
                    "prob: a closed form was computed and withheld — no admissible \
                     parameter point could be constructed to check it at"
                ),
            },
            ProbError::Divergent(msg) => write!(f, "prob: the quantity does not exist: {msg}"),
        }
    }
}

impl std::error::Error for ProbError {}

impl AlkahestError for ProbError {
    fn code(&self) -> &'static str {
        match self {
            ProbError::InvalidParameter { .. } => "E-PROB-001",
            ProbError::Unsupported(_) => "E-PROB-002",
            ProbError::IntegralDidNotClose(_) => "E-PROB-003",
            ProbError::NoClosedForm { .. } => "E-PROB-004",
            ProbError::Unverified(_) => "E-PROB-005",
            ProbError::Divergent(_) => "E-PROB-006",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        match self {
            ProbError::InvalidParameter { .. } => Some(
                "pass a parameter inside the distribution's constraint, or leave it symbolic \
                 (a symbolic parameter is carried as a side condition on `constraints` \
                 rather than decided)",
            ),
            ProbError::Unsupported(_) => Some(
                "reduce the query to a supported shape: a polynomial or exponential `f` for \
                 `expectation`, an affine combination for `expectation_affine`",
            ),
            ProbError::IntegralDidNotClose(_) => Some(
                "the named integral has no antiderivative this integrator found; integrate it \
                 numerically, or ask for a moment/CDF that is in the closed-form table",
            ),
            ProbError::NoClosedForm { .. } => Some(
                "the closed form needs a special function this library does not implement; \
                 use an integer shape parameter where that collapses the special function to \
                 a finite sum (Erlang, integer-parameter Beta), or integrate numerically",
            ),
            ProbError::Unverified(_) => Some(
                "an unconfirmed closed form is withheld rather than returned; report the \
                 distribution and query as a minimal failing example",
            ),
            ProbError::Divergent(_) => Some(
                "the integral defining this expectation does not converge — there is no value \
                 to return; restrict `f` so that `f(x)·p(x)` is integrable over the support",
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Support
// ---------------------------------------------------------------------------

/// Where a distribution puts its mass.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Support {
    /// The whole real line.
    Real,
    /// `(0, ∞)`.
    Positive,
    /// The closed interval `[lo, hi]`, endpoints symbolic.
    Interval(ExprId, ExprId),
    /// `{0, 1, …, n}`.
    IntegersUpTo(u32),
    /// `{0, 1, 2, …}` — countably infinite.
    NonNegativeIntegers,
}

impl Support {
    /// Is this a discrete support?
    pub fn is_discrete(&self) -> bool {
        matches!(
            self,
            Support::IntegersUpTo(_) | Support::NonNegativeIntegers
        )
    }
}

// ---------------------------------------------------------------------------
// Distribution
// ---------------------------------------------------------------------------

/// Which distribution. Parameters live on [`Distribution`], not here.
///
/// `#[non_exhaustive]`: adding a distribution must not be a semver break.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DistKind {
    /// `Normal(μ, σ)`, `σ > 0`, density `e^{-(x-μ)²/(2σ²)} / (σ√(2π))` on `ℝ`.
    Normal,
    /// `LogNormal(μ, σ)`, `σ > 0`: `log X ~ Normal(μ, σ)`. Density
    /// `e^{-(log x - μ)²/(2σ²)} / (x σ√(2π))` on `(0, ∞)`.
    LogNormal,
    /// `Uniform(a, b)`, `a < b`, density `1/(b-a)` on `[a, b]`.
    Uniform,
    /// `Exponential(λ)`, `λ > 0`, density `λ e^{-λx}` on `(0, ∞)`.
    Exponential,
    /// `Gamma(k, θ)`, `k > 0`, `θ > 0` — **shape–scale**, density
    /// `x^{k-1} e^{-x/θ} / (Γ(k) θ^k)` on `(0, ∞)`.
    Gamma,
    /// `Beta(α, β)`, `α > 0`, `β > 0`, density
    /// `x^{α-1}(1-x)^{β-1} / B(α, β)` on `[0, 1]`.
    Beta,
    /// `Bernoulli(p)`, `0 ≤ p ≤ 1`, on `{0, 1}`.
    Bernoulli,
    /// `Binomial(n, p)`, `n` a non-negative integer, `0 ≤ p ≤ 1`, on `{0…n}`.
    Binomial,
    /// `Poisson(λ)`, `λ > 0`, on `{0, 1, 2, …}`.
    Poisson,
}

impl DistKind {
    /// The distribution's name, as it is printed and as it appears in
    /// derivation logs.
    pub fn name(self) -> &'static str {
        match self {
            DistKind::Normal => "Normal",
            DistKind::LogNormal => "LogNormal",
            DistKind::Uniform => "Uniform",
            DistKind::Exponential => "Exponential",
            DistKind::Gamma => "Gamma",
            DistKind::Beta => "Beta",
            DistKind::Bernoulli => "Bernoulli",
            DistKind::Binomial => "Binomial",
            DistKind::Poisson => "Poisson",
        }
    }
}

/// A distribution with symbolic parameters.
///
/// Constructed through the named constructors ([`Distribution::normal`],
/// [`Distribution::gamma`], …), which is what makes it impossible to hold one
/// whose *numeric* parameters violate its own constraints. Symbolic parameters
/// are not decidable and are carried on [`Distribution::constraints`] instead.
///
/// Fields are private: a publicly-constructible struct cannot gain a field
/// without a semver break, and this one will gain fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Distribution {
    kind: DistKind,
    params: Vec<ExprId>,
}

impl Distribution {
    /// Which distribution this is.
    pub fn kind(&self) -> DistKind {
        self.kind
    }

    /// The parameters, in the order of the constructor.
    pub fn params(&self) -> &[ExprId] {
        &self.params
    }

    /// `Normal(μ, σ)` — requires `σ > 0`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `σ` is a number that is not
    /// positive. A symbolic `σ` is accepted and carried on
    /// [`Distribution::constraints`].
    pub fn normal(mu: ExprId, sigma: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(sigma, "sigma", pool)?;
        Ok(Distribution {
            kind: DistKind::Normal,
            params: vec![mu, sigma],
        })
    }

    /// `LogNormal(μ, σ)` — `log X ~ Normal(μ, σ)`, requires `σ > 0`.
    ///
    /// Note the convention: `μ` and `σ` are the parameters of the *underlying
    /// normal*, not the mean and standard deviation of `X`. `E[X] =
    /// e^{μ+σ²/2}`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `σ` is a non-positive number.
    pub fn log_normal(mu: ExprId, sigma: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(sigma, "sigma", pool)?;
        Ok(Distribution {
            kind: DistKind::LogNormal,
            params: vec![mu, sigma],
        })
    }

    /// `Uniform(a, b)` — requires `a < b`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `a` and `b` are both numbers and
    /// `a ≥ b`.
    pub fn uniform(a: ExprId, b: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_less(a, b, "b", "a < b", pool)?;
        Ok(Distribution {
            kind: DistKind::Uniform,
            params: vec![a, b],
        })
    }

    /// `Exponential(λ)` — rate parametrisation, mean `1/λ`, requires `λ > 0`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `λ` is a non-positive number.
    pub fn exponential(lambda: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(lambda, "lambda", pool)?;
        Ok(Distribution {
            kind: DistKind::Exponential,
            params: vec![lambda],
        })
    }

    /// `Gamma(k, θ)` — **shape–scale**, mean `kθ`, requires `k > 0`, `θ > 0`.
    ///
    /// The other common convention is shape–*rate*; pass `1/rate` as `θ`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when either parameter is a non-positive
    /// number.
    pub fn gamma(k: ExprId, theta: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(k, "k", pool)?;
        dists::require_positive(theta, "theta", pool)?;
        Ok(Distribution {
            kind: DistKind::Gamma,
            params: vec![k, theta],
        })
    }

    /// `Beta(α, β)` — requires `α > 0`, `β > 0`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when either parameter is a non-positive
    /// number.
    pub fn beta(alpha: ExprId, beta: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(alpha, "alpha", pool)?;
        dists::require_positive(beta, "beta", pool)?;
        Ok(Distribution {
            kind: DistKind::Beta,
            params: vec![alpha, beta],
        })
    }

    /// `Bernoulli(p)` — requires `0 ≤ p ≤ 1`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `p` is a number outside `[0, 1]`.
    pub fn bernoulli(p: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_probability(p, "p", pool)?;
        Ok(Distribution {
            kind: DistKind::Bernoulli,
            params: vec![p],
        })
    }

    /// `Binomial(n, p)` — `n` trials, requires `0 ≤ p ≤ 1`.
    ///
    /// `n` must be a literal non-negative integer: the support `{0…n}` is
    /// enumerated by every route in this module, so a symbolic `n` is not a
    /// distribution this module can compute with and is refused rather than
    /// half-supported.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `n` is not a literal non-negative
    /// integer, or `p` is a number outside `[0, 1]`.
    pub fn binomial(n: ExprId, p: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_count(n, "n", pool)?;
        dists::require_probability(p, "p", pool)?;
        Ok(Distribution {
            kind: DistKind::Binomial,
            params: vec![n, p],
        })
    }

    /// `Poisson(λ)` — requires `λ > 0`.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] when `λ` is a non-positive number.
    pub fn poisson(lambda: ExprId, pool: &ExprPool) -> Result<Self, ProbError> {
        dists::require_positive(lambda, "lambda", pool)?;
        Ok(Distribution {
            kind: DistKind::Poisson,
            params: vec![lambda],
        })
    }

    /// Where the mass lives.
    pub fn support(&self, pool: &ExprPool) -> Support {
        dists::support(self, pool)
    }

    /// The parameter constraints, as predicate expressions.
    ///
    /// A constraint whose parameters are numeric has already been *checked* by
    /// the constructor, so it appears here as a record rather than an
    /// obligation. A constraint over a symbolic parameter has **not** been
    /// decided and is the caller's to discharge — that is the "carried as a
    /// side condition" half of the contract, and it is why this returns
    /// predicates rather than a bool.
    pub fn constraints(&self, pool: &ExprPool) -> Vec<ExprId> {
        dists::constraints(self, pool)
    }

    /// The density at `x` (continuous) or the mass at `k` (discrete).
    ///
    /// For a continuous distribution this is the density *on the support only*
    /// — it is not extended by zero outside it, because a `Piecewise` density
    /// defeats every downstream symbolic route for no gain. Use
    /// [`Distribution::support`] to know where it is valid.
    pub fn pdf(&self, x: ExprId, pool: &ExprPool) -> ExprId {
        dists::pdf(self, x, pool)
    }

    /// `E[X]`, in closed form, verified.
    ///
    /// # Errors
    ///
    /// See [`ProbError`]; in particular [`ProbError::Unverified`] if the
    /// numeric gate could not confirm the table value against quadrature.
    pub fn mean(&self, pool: &ExprPool) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        moments::mean(self, pool)
    }

    /// `Var[X] = E[X²] - E[X]²`, in closed form, verified.
    ///
    /// # Errors
    ///
    /// See [`ProbError`].
    pub fn variance(
        &self,
        pool: &ExprPool,
    ) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        moments::variance(self, pool)
    }

    /// The raw moment `E[Xⁿ]`, in closed form, verified.
    ///
    /// # Errors
    ///
    /// See [`ProbError`]. `n` past [`MAX_MOMENT_ORDER`] is
    /// [`ProbError::Unsupported`]: the closed forms stay exact but the numeric
    /// gate loses its ability to *fail*, and an unfailable check is decoration.
    pub fn moment(
        &self,
        n: u32,
        pool: &ExprPool,
    ) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        moments::moment(self, n, pool)
    }

    /// `P(X ≤ x)`, in closed form, verified.
    ///
    /// # Outside the support
    ///
    /// Unlike [`Distribution::pdf`], this is **not** "the formula on the
    /// support, and you are on your own elsewhere". `P(X ≤ x)` is `0` below the
    /// support and `1` above it, and the in-support closed form does not merely
    /// fail there — it produces a clean wrong number offered as a probability
    /// (the `Erlang(3, 4/5)` form at `x = -5` is `-7396.87`). So:
    ///
    /// * an argument that can be *decided* to sit outside the support returns
    ///   the exact `0` or `1`;
    /// * an argument that cannot be decided returns the in-support branch with
    ///   the restriction recorded as a [`crate::deriv::SideCondition`] on the
    ///   `prob_cdf_argument_inside_support` step, rather than silently assumed.
    ///
    /// The second case is a weaker guarantee than the first, and it is the one
    /// a symbolic argument gets.
    ///
    /// # Errors
    ///
    /// [`ProbError::NoClosedForm`] where the CDF needs a special function this
    /// library does not have — the incomplete gamma for a non-integer `Gamma`
    /// shape, the incomplete beta for non-integer `Beta` parameters.
    pub fn cdf(
        &self,
        x: ExprId,
        pool: &ExprPool,
    ) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        moments::cdf(self, x, pool)
    }

    /// The characteristic function `φ_X(t) = E[e^{itX}]`, in closed form,
    /// verified — against the defining integral **and**, where
    /// [`crate::transform::fourier_transform`] has a rule for the density,
    /// against `F{p}(-t/2π)`.
    ///
    /// `φ` is a table entry keyed on the distribution, not a transform of
    /// [`Distribution::pdf`]: a law can have a closed-form `φ` and no
    /// closed-form density at all, and a design that derives one from the other
    /// cannot express that. See [`crate::prob::characteristic_function`] — in
    /// particular its note on **inversion**, of which there is none here.
    ///
    /// # Errors
    ///
    /// [`ProbError::NoClosedForm`] for `LogNormal` (whose `φ` has no closed
    /// form at all) and `Beta` (which needs `₁F₁`).
    pub fn characteristic_function(
        &self,
        t: ExprId,
        pool: &ExprPool,
    ) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        charfun::characteristic_function(self, t, pool)
    }

    /// The quantile `F⁻¹(p)`, in closed form, verified.
    ///
    /// `p` is a probability and is checked as one: a numeric `p` outside
    /// `[0, 1]` is [`ProbError::InvalidParameter`], because the closed forms
    /// return a number there rather than failing — `Uniform(-2, 3)` at `p = 2`
    /// evaluates to `8`, outside the support it claims to be a point of. A
    /// symbolic `p` cannot be decided and carries `0 ≤ p ≤ 1` as a
    /// [`crate::deriv::SideCondition`] instead.
    ///
    /// # Errors
    ///
    /// [`ProbError::InvalidParameter`] for a numeric `p` outside `[0, 1]`, and
    /// [`ProbError::NoClosedForm`] for every distribution whose quantile needs
    /// `erf⁻¹` or a numerical inversion — which is most of them. `Uniform` and
    /// `Exponential` close; nothing else here does.
    pub fn quantile(
        &self,
        p: ExprId,
        pool: &ExprPool,
    ) -> Result<crate::deriv::DerivedExpr<ExprId>, ProbError> {
        moments::quantile(self, p, pool)
    }
}

/// Highest moment order the numeric gate can still *fail* on, and therefore
/// the highest [`Distribution::moment`] will return.
///
/// Not a capability limit — a verification limit. `E[X¹⁶]` under a unit normal
/// is `2027025`; the quadrature that checks it integrates `z¹⁶ e^{-z²/2}`,
/// whose mass sits where `z¹⁶` spans twenty orders of magnitude, and the
/// double-exponential rule's relative error there stops being small enough to
/// distinguish the right answer from a neighbouring wrong one. Past this the
/// check would pass unconditionally, so the moment is refused instead of
/// returned unchecked.
pub const MAX_MOMENT_ORDER: u32 = 12;

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// `π`, the way the rest of the crate spells it.
pub(crate) fn pi(pool: &ExprPool) -> ExprId {
    pool.symbol("pi", Domain::Real)
}

/// `a - b`.
pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let neg = pool.integer(-1);
    pool.add(vec![a, pool.mul(vec![neg, b])])
}

/// `a / b`.
pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let inv = pool.pow(b, pool.integer(-1));
    pool.mul(vec![a, inv])
}
