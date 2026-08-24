//! Continuous creative telescoping: **Almkvist–Zeilberger**, the differential
//! twin of [`mod@super::zeilberger`].
//!
//! # Scope
//!
//! Where the discrete engine takes a proper hypergeometric summand `F(n,k)`,
//! the shift `k → k+1` and the difference operator `Δ_k`, this one takes a
//! *hyperexponential* integrand `F(n,x)`, the derivation `D_x`, and produces
//! the same shape of certificate:
//!
//! ```text
//! Σ_{i=0}^{J} a_i(n)·F(n+i, x) = D_x( R(n,x)·F(n,x) )
//! ```
//!
//! with polynomial coefficients `a_i(n)` (not all zero, `a_J ≢ 0`) and an exact
//! rational certificate `R ∈ Q(n)(x)`. That identity — and only that identity —
//! is what [`almkvist_zeilberger`] verifies exactly before returning.
//!
//! | discrete ([`mod@super::zeilberger`]) | continuous (here) |
//! |---|---|
//! | `Σ_k F(n,k)` | `∫_a^b F(n,x) dx` |
//! | shift `k → k+1`, `Δ_k` | derivation `D_x` |
//! | proper hypergeometric term | hyperexponential term ([`mod@hyperexp`]) |
//! | Gosper's algorithm | differential Gosper ([`mod@dgosper`]) |
//! | [`mod@super::boundary`] | [`mod@boundary`] |
//!
//! # Module layout
//!
//! - [`mod@hyperexp`] — recognition of
//!   `F(n,x) = R(n,x)·wⁿ·exp(η(x))·∏ B_j(x)^(α_j n + β_j)` and its two exact
//!   certificates: the logarithmic derivative `θ = ∂_x F / F ∈ Q(n)(x)` and the
//!   shift ratios `F(n+i,x)/F(n,x) ∈ Q(n)(x)`.
//! - [`mod@rde`] — the shared inner solver. Both stages below are the *same*
//!   parametric Risch differential equation `R' + θ·R = Σ_i a_i·r_i` solved by
//!   undetermined coefficients over `Q(n)`; the only difference is whether the
//!   right-hand side is fixed (`J = 0`, `a_0 = 1`) or carries unknowns.
//! - [`mod@dgosper`] — the indefinite case: decide whether `∫F dx = R·F` for a
//!   rational `R`, and return it. This is `J = 0` with `a_0` normalised to `1`.
//! - [`mod@search`] — [`almkvist_zeilberger`] itself, ascending in `J` from `0`.
//! - [`mod@boundary`] — the continuous analogue of [`mod@super::boundary`]:
//!   deciding whether `[R·F]_a^b` vanishes, so that the certificate becomes a
//!   recurrence for `f(n) = ∫_a^b F(n,x) dx` rather than a statement about the
//!   integrand alone.
//!
//! # The integral recurrence carries a hypothesis
//!
//! Integrating the certificate over `x ∈ [a,b]` gives
//!
//! ```text
//! Σ_i a_i(n)·f(n+i) = [ R(n,x)·F(n,x) ]_a^b,     f(n) = ∫_a^b F(n,x) dx
//! ```
//!
//! so `Σ_i a_i(n)·f(n+i) = 0` holds **only when that boundary term vanishes**.
//! The certificate says nothing about it: `R·F` is a perfectly good
//! antiderivative of the left-hand side whether or not it happens to vanish at
//! the limits, and whether or not the integral converges there at all.
//! [`boundary::integral_boundary_status`] decides it three-valued, in the same
//! discipline as the discrete module — see its docs, and do not read
//! [`boundary::IntegralBoundaryStatus::Unknown`] as a vanishing boundary.
//!
//! # Honest limitations (read before relying on this)
//!
//! - **Integrand class**: hyperexponential in `x` *and* hypergeometric in `n`,
//!   in the specific shape [`hyperexp::HyperExpTerm`] parses — a rational
//!   `Q(n)(x)` prefactor times `wⁿ` (`w ∈ Q`) times `exp(η)` with `η ∈ Q(n)(x)`
//!   times finitely many `B(x)^(α·n + β)` with `α ∈ Z`, `β ∈ Q`. Anything
//!   outside it — `log x`, `sin x`, a *sum* of two hyperexponential terms, an
//!   algebraic function that is not a rational power — is refused as
//!   [`DiffTelescopingError::NotHyperexponential`], never approximated.
//!   `exp(n·x)` is the instructive refusal: it *is* hyperexponential in `x`,
//!   but `F(n+1,x)/F(n,x) = eˣ` is not rational, so no algorithm in this family
//!   applies and [`DiffTelescopingError::NotHypergeometricInN`] says so
//!   specifically rather than reporting a shape failure.
//! - **Formal, not analytic**: `B(x)^β` for non-integer `β` is treated as a
//!   formal symbol whose logarithmic derivative is `β·B'/B`. The certificate is
//!   an identity of formal hyperexponential expressions; reading it as an
//!   identity of *functions* additionally requires a branch of `B^β` to be
//!   fixed consistently on the interval in question. That is a real hypothesis
//!   on e.g. `x^(1/2)` across `x = 0`, and it is stated rather than assumed.
//! - **Certificate denominator**: the ansatz is `R = P(x)/(D(x)^κ·B(x))` with
//!   `D` the denominator of `θ`, `B` the common denominator of the shift ratios,
//!   `deg P ≤ d` and `κ ≤ max_den_power`. The *support* of that denominator is
//!   not a guess — a pole of `R` must be a pole of `θ` or of the right-hand
//!   side, which is exactly `D` and `B` — but the *multiplicity* `κ` is a
//!   bounded search, and one specific case pushes it up without bound: at a
//!   simple pole of `θ` with residue `ρ`, `R` may have a pole of order `ρ`
//!   whenever `ρ` is a positive integer (`∫x²·eˣ dx = (x²−2x+2)·eˣ` needs
//!   `κ = 2`). With a *symbolic* `ρ ∈ Q(n)` — the usual case for `xⁿ·…` —
//!   whether that resonance occurs is not decidable at all, and this module
//!   does not pretend otherwise: it searches `κ` and refuses with
//!   [`DiffTelescopingError::SearchExhausted`] when the bound runs out.
//! - **Minimal order, not minimal degree**: [`mod@search`] ascends in `J` from `0`
//!   and tries every `(κ, d)` in bounds before moving on, so a returned order
//!   *is* the least one reachable within `max_den_power`/`max_degree`. It is
//!   not the cost-ordered iterative deepening [`mod@super::zeilberger`] uses, so
//!   raising `max_degree` is not free the way it is there.
//! - **Boundary**: only constant (`n`-independent) limits are analysed, and
//!   only the endpoint kinds [`mod@boundary`] documents. Convergence conditions on
//!   `n` are *reported*, not silently assumed — see
//!   [`boundary::IntegralBoundaryStatus::side_conditions`].
//! - **Rust-internal**: there is no PyO3 binding for any of this yet.
//!
//! Every certificate returned by any entry point in this module has been
//! re-derived and checked as an exact identity in `Q(n)(x)` first, exactly as
//! [`super::zeilberger::zeilberger()`] does. A successful call is a proof; an
//! unverifiable candidate is discarded and the search continues.

pub mod boundary;
pub mod dgosper;
pub mod hyperexp;
pub mod rde;
pub mod search;

pub use boundary::{integral_boundary_status, IntegralBoundaryStatus, IntegrationLimit};
pub use dgosper::{dgosper, dgosper_term};
pub use hyperexp::{hyperexp_log_derivative, HyperExpTerm, PowerFactor};
pub use search::{almkvist_zeilberger, integrand_antiderivative, AzResult};

/// Search bounds for the continuous engine.
///
/// All three are upper *bounds*, not starting points: the search walks the
/// `(order, κ, degree)` grid from the cheapest corner, so raising any of them
/// only widens what can be found. Unlike [`mod@super::zeilberger`]'s
/// cost-ordered deepening, though, the walk here is plain ascending nested
/// loops, so raising `max_degree` genuinely does cost more on an input that
/// has no certificate at all.
#[derive(Debug, Clone, Copy)]
pub struct AzOpts {
    /// Largest recurrence order `J` to try. Orders are searched from `0`
    /// upward, so a returned order is the least one reachable within the other
    /// two bounds. Ignored by [`dgosper()`], where the order is `0` by
    /// definition.
    pub max_order: usize,
    /// Largest degree in `x` of the certificate numerator `P`.
    pub max_degree: usize,
    /// Largest power `κ` of `θ`'s denominator admitted in the certificate
    /// denominator `D^κ·B`. See [`mod@rde`] for what this bound is and is not:
    /// the denominator's *support* is forced, only its multiplicity is
    /// searched.
    pub max_den_power: usize,
}

impl Default for AzOpts {
    fn default() -> Self {
        AzOpts {
            max_order: 4,
            max_degree: 12,
            max_den_power: 3,
        }
    }
}

use std::fmt;

/// Errors from the continuous (differential) creative-telescoping engine.
///
/// A separate type from [`super::HolonomicError`] for the same reason
/// [`super::telescoping2d::Telescoping2dError`] is: the public enums in this
/// subsystem are exhaustive, so sharing one would make every new engine a
/// major-version break, and a caller could not tell which engine refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffTelescopingError {
    /// The integrand is not hyperexponential in `x` in the shape
    /// [`hyperexp::HyperExpTerm`] recognises.
    NotHyperexponential(String),
    /// The integrand *is* hyperexponential in `x`, but `F(n+i,x)/F(n,x)` is not
    /// a rational function of `x` — so creative telescoping in `n` does not
    /// apply, however well-behaved the `x` side is. `exp(n·x)` is the canonical
    /// case.
    NotHypergeometricInN(String),
    /// The bounded ansatz search (order, certificate degree, denominator power)
    /// was exhausted without finding a certificate that passed exact
    /// verification.
    SearchExhausted(String),
    /// A candidate was found by the search but failed the exact `Q(n)(x)`
    /// identity check, and no other candidate succeeded. Refused rather than
    /// returned unverified.
    CertificateVerificationFailed(String),
    /// Malformed call (`n` and `x` not distinct, degenerate bounds, …).
    InvalidInput(String),
}

impl fmt::Display for DiffTelescopingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffTelescopingError::NotHyperexponential(s) => {
                write!(f, "almkvist-zeilberger: not a hyperexponential term: {s}")
            }
            DiffTelescopingError::NotHypergeometricInN(s) => {
                write!(
                    f,
                    "almkvist-zeilberger: not hypergeometric in the outer index: {s}"
                )
            }
            DiffTelescopingError::SearchExhausted(s) => {
                write!(f, "almkvist-zeilberger: search exhausted: {s}")
            }
            DiffTelescopingError::CertificateVerificationFailed(s) => {
                write!(
                    f,
                    "almkvist-zeilberger: certificate failed exact verification: {s}"
                )
            }
            DiffTelescopingError::InvalidInput(s) => {
                write!(f, "almkvist-zeilberger: invalid input: {s}")
            }
        }
    }
}

impl std::error::Error for DiffTelescopingError {}

impl crate::errors::AlkahestError for DiffTelescopingError {
    fn code(&self) -> &'static str {
        match self {
            DiffTelescopingError::NotHyperexponential(_) => "E-HOLO-060",
            DiffTelescopingError::NotHypergeometricInN(_) => "E-HOLO-061",
            DiffTelescopingError::SearchExhausted(_) => "E-HOLO-062",
            DiffTelescopingError::CertificateVerificationFailed(_) => "E-HOLO-063",
            DiffTelescopingError::InvalidInput(_) => "E-HOLO-064",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            DiffTelescopingError::NotHyperexponential(_) => {
                "write the integrand as R(n,x)*w**n*exp(eta(x))*prod(B(x)**(a*n + b)) with \
                 integer a and rational b; supported function heads are exp and sqrt, and a \
                 sum of two hyperexponential terms is not itself hyperexponential"
            }
            DiffTelescopingError::NotHypergeometricInN(_) => {
                "the x-side is fine but F(n+1,x)/F(n,x) is not rational in x — e.g. exp(n*x), \
                 whose ratio is exp(x). No algorithm in this family applies; close the branch"
            }
            DiffTelescopingError::SearchExhausted(_) => {
                "raise max_order, max_degree and/or max_den_power in AzOpts; if the integrand \
                 genuinely has no such certificate within reach — or needs a certificate \
                 denominator this module's D**kappa * B ansatz cannot represent — this method \
                 does not apply"
            }
            DiffTelescopingError::CertificateVerificationFailed(_) => {
                "internal: report the integrand as a minimal failing example"
            }
            DiffTelescopingError::InvalidInput(_) => {
                "n and x must be distinct symbols; the max_order, max_degree and max_den_power \
                 bounds must be within the documented ranges"
            }
        })
    }
}
