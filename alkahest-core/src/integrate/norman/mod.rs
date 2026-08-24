//! Risch–Norman ("parallel Risch") heuristic integration.
//!
//! # What this is
//!
//! Instead of building a differential-field tower and recursing on it — what
//! [`crate::integrate::risch`] does — the parallel-Risch heuristic *posits* the
//! shape of the answer,
//!
//! ```text
//!     F  =  Σᵢ cᵢ·mᵢ  +  Σⱼ dⱼ·log(pⱼ)
//! ```
//!
//! differentiates it, and equates the result against the integrand.  The
//! unknown constants `cᵢ`, `dⱼ` enter linearly, so the whole problem collapses
//! to **one linear system over `ℚ`**.  The method is due to Norman & Moore
//! (1977); Davenport, *The parallel Risch algorithm (I)* (1982) is the standard
//! write-up, and Geddes, Czapor & Labahn ch. 12 the textbook treatment.
//!
//! It is the default integrator in Maple (`int`), Axiom/FriCAS (Norman–Moore)
//! and Reduce, with full Risch behind it, because it is cheap and it closes a
//! large fraction of everyday integrands without any tower construction.
//!
//! # The contract — a decline is not a verdict
//!
//! [`integrate_parallel_risch`] returns [`ParallelRischOutcome`], which has
//! exactly two shapes: [`ParallelRischOutcome::Solved`] carries a
//! **verification-gated** antiderivative, and
//! [`ParallelRischOutcome::Declined`] carries a [`DeclineReason`].
//!
//! **There is deliberately no variant that a caller could read as a proof of
//! non-elementarity.**  A Risch–Norman failure means "my ansatz did not
//! close" — the monomial box was too small, the logarithm candidate set did
//! not contain the right argument, the constant field was too small (`ℚ`
//! only — see the limitations below), or the structure-theorem precondition
//! could not be certified.  `∫exp(x²)dx` and `∫dx/(x²+1)` both decline here;
//! the first has no elementary antiderivative and the second is `arctan x`.
//! Nothing in the return type distinguishes them, because this module cannot.
//! [`DeclineReason::into_integration_error`] therefore maps *every* decline to
//! [`IntegrationError::NotImplemented`], never to `NonElementary`.
//!
//! (This is the pattern that `solve_rational_rde_generalized` got wrong: its
//! `None` was documented as certifying non-elementarity and was converted
//! straight into a certificate.  A heuristic's silence is not evidence.)
//!
//! # Soundness
//!
//! Two independent guards:
//!
//! 1. **Algebraic independence up front.**  Equating coefficients of monomials
//!    in the generators is only valid when those generators are algebraically
//!    independent over `ℚ(x)` (Bronstein, *Structure theorems for parallel
//!    integration*, JSC 42(7):757–769, 2007).  [`ring`] reduces exponential
//!    arguments to a `ℤ`-lattice basis and checks logarithm arguments for
//!    multiplicative independence modulo constants, and declines when it
//!    cannot certify the condition.  See that module for the details.
//! 2. **`d/dx F = f` before return.**  Every candidate is passed through
//!    [`crate::integrate::verify_antiderivative_status`] against the *original*
//!    integrand.  A candidate that does not verify is discarded and the call
//!    declines.  The construction is sound by design; the gate is there
//!    because construction arguments have bugs.
//!
//! A singular or inconsistent linear system declines
//! ([`DeclineReason::NoSolution`]); it never produces a partial answer.
//!
//! # Coverage
//!
//! Integrands that are **rational functions of `x` and of exp/log
//! generators**, whose antiderivative is a rational function of the same
//! generators plus `ℚ`-multiples of logarithms of ring elements.  In practice
//! that means:
//!
//! * everything the rational-function integrator gets without needing
//!   `RootSum` or `arctan`;
//! * mixed exp/log integrands where tower construction is the hard part but
//!   the answer's shape is simple — `∫exp(x)/(exp(x)+1)`,
//!   `∫exp(2x)/(exp(x)+1)`, `∫dx/(1+exp(−x))`, `∫dx/(x·log x)`;
//! * polynomial-times-exponential and polynomial-times-logarithm families.
//!
//! # Limitations
//!
//! In the register of [`crate::holonomic`]: this list is what the module
//! *cannot* do, stated so that a decline is never mistaken for a verdict.
//!
//! * **The constant field is `ℚ`.**  Answers needing `ℚ(i)` (`arctan`,
//!   `∫dx/(x²+1)`), a real quadratic field (`∫√2`-scaled logarithms, the
//!   classical `∫√(tan x)`), or algebraic residues (`RootSum`, which the
//!   existing rational integrator does produce) are out of reach.  This is the
//!   single biggest source of declines on textbook input.
//! * **No trigonometric, inverse-trigonometric, hyperbolic or algebraic
//!   generators.**  The ring is `ℚ(x, exp …, log …)` only.  `sin`, `cos`,
//!   `tan`, `atan`, `sqrt` and any non-integer power decline immediately.
//!   Trigonometric coverage would need either a `ℚ(i)` constant field with
//!   complex exponentials or explicit `tan` monomials; neither is implemented.
//! * **The ansatz is a box, not a bound.**  The monomial degrees are bounded
//!   heuristically by `deg(N) + deg(Q) + 1` per variable, and the logarithm
//!   arguments are drawn from the irreducible factors of the denominator
//!   together with `x` and the tower's logarithm generators.  An elementary
//!   antiderivative outside that set is not found.  No claim of completeness
//!   is made or intended — that is what the full Risch algorithm is for.
//! * **The independence checks are conservative.**  A tower whose generators
//!   *are* independent but which the checks cannot certify is declined rather
//!   than attempted.  `∫f(log 2x, log x)` is the canonical example.
//! * **Size caps.**  The ansatz box, the common denominator and the linear
//!   system are capped ([`ring::MAX_GENERATORS`] and the constants in
//!   [`ansatz`]); exceeding any of them declines with
//!   [`DeclineReason::TooLarge`] rather than running to exhaustion.
//! * **It is not wired into [`crate::integrate::integrate`].**  This is a
//!   separate entry point on purpose, so its coverage can be measured against
//!   the existing engine before any routing change is made.

use std::fmt;

use crate::integrate::{
    verify_antiderivative_status, AntiderivativeVerification, IntegrationError,
};
use crate::kernel::{ExprId, ExprPool};

pub(crate) mod ansatz;
pub(crate) mod ring;

#[cfg(test)]
mod tests;

/// Why the heuristic declined.
///
/// **None of these is a statement about the integrand's integrability.**  See
/// the module documentation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeclineReason {
    /// The integrand is outside the ring `ℚ(x, exp …, log …)` — a
    /// trigonometric function, a radical, a foreign symbol, a float.
    UnsupportedIntegrand(String),
    /// The generators could not be certified algebraically independent, so
    /// equating coefficients of monomials would not be meaningful.
    DependentGenerators(&'static str),
    /// The linear system was inconsistent, or the ansatz simply did not
    /// contain the answer.  **Not** a proof that no answer exists.
    NoSolution,
    /// A size cap was hit before the linear algebra could run.
    TooLarge(&'static str),
    /// Exact arithmetic in the ring failed (a zero denominator, a FLINT
    /// factorisation that did not succeed, …).
    RingArithmetic,
    /// A candidate was produced but `d/dx F = f` could not be confirmed, so it
    /// was discarded.
    VerificationFailed,
}

impl fmt::Display for DeclineReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeclineReason::UnsupportedIntegrand(what) => {
                write!(f, "integrand is outside the exp/log ring: {what}")
            }
            DeclineReason::DependentGenerators(what) => {
                write!(f, "generators are not certifiably independent: {what}")
            }
            DeclineReason::NoSolution => write!(
                f,
                "the Risch–Norman ansatz did not close (this is not a \
                 non-elementarity result)"
            ),
            DeclineReason::TooLarge(what) => write!(f, "{what} exceeded the size cap"),
            DeclineReason::RingArithmetic => write!(f, "exact ring arithmetic failed"),
            DeclineReason::VerificationFailed => {
                write!(f, "the candidate antiderivative failed the d/dx F = f gate")
            }
        }
    }
}

impl DeclineReason {
    /// Map a decline onto the integrator's error type.
    ///
    /// This is deliberately total and deliberately boring: **every** decline
    /// becomes [`IntegrationError::NotImplemented`].  There is no path from
    /// this module to [`IntegrationError::NonElementary`], because the
    /// heuristic cannot establish non-elementarity.
    pub fn into_integration_error(self) -> IntegrationError {
        IntegrationError::NotImplemented(format!("risch-norman: {self}"))
    }
}

/// The result of a Risch–Norman attempt.
///
/// Note what is *absent*: there is no variant meaning "no elementary
/// antiderivative exists".  A [`ParallelRischOutcome::Declined`] can only ever
/// be read as "this heuristic did not find one".
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ParallelRischOutcome {
    /// An antiderivative that passed the `d/dx F = f` gate.
    Solved {
        /// The antiderivative `F`, simplified.
        antiderivative: ExprId,
        /// How `d/dx F = f` was established.
        verification: AntiderivativeVerification,
    },
    /// The heuristic did not produce a verified antiderivative.
    Declined(DeclineReason),
}

impl ParallelRischOutcome {
    /// The antiderivative, if one was found and verified.
    pub fn antiderivative(&self) -> Option<ExprId> {
        match self {
            ParallelRischOutcome::Solved { antiderivative, .. } => Some(*antiderivative),
            ParallelRischOutcome::Declined(_) => None,
        }
    }

    /// `true` when an antiderivative was found and verified.
    pub fn is_solved(&self) -> bool {
        matches!(self, ParallelRischOutcome::Solved { .. })
    }
}

/// Integrate `expr` with respect to `var` using the Risch–Norman heuristic.
///
/// Returns a verification-gated antiderivative or a [`DeclineReason`].  Read
/// the module documentation before wiring this into anything: a decline is not
/// a non-elementarity verdict.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool};
/// use alkahest_cas::integrate::norman::{integrate_parallel_risch, ParallelRischOutcome};
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// // ∫ exp(x)/(exp(x) + 1) dx = log(exp(x) + 1)
/// let e = pool.func("exp", vec![x]);
/// let den = pool.add(vec![e, pool.integer(1_i32)]);
/// let f = pool.mul(vec![e, pool.pow(den, pool.integer(-1_i32))]);
/// assert!(matches!(
///     integrate_parallel_risch(f, x, &pool),
///     ParallelRischOutcome::Solved { .. }
/// ));
/// ```
pub fn integrate_parallel_risch(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> ParallelRischOutcome {
    match attempt(expr, var, pool) {
        Ok(candidate) => match verify_antiderivative_status(candidate, expr, var, pool) {
            Some(verification) => ParallelRischOutcome::Solved {
                antiderivative: candidate,
                verification,
            },
            None => ParallelRischOutcome::Declined(DeclineReason::VerificationFailed),
        },
        Err(reason) => ParallelRischOutcome::Declined(reason),
    }
}

/// The unverified half of [`integrate_parallel_risch`].
fn attempt(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, DeclineReason> {
    let normalized = crate::simplify::engine::simplify(expr, pool).value;
    let ring = ring::build(normalized, var, pool)?;
    let f = ring.to_rf(normalized, pool)?;
    if f.numer.is_zero() {
        return Ok(pool.integer(0_i32));
    }
    ansatz::solve(&ring, &f, pool)
}
