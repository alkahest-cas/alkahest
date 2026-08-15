//! Holonomic (D-finite) machinery: creative telescoping, Zeilberger's
//! algorithm, and exact `Q(n)` / `Q(n)(k)` arithmetic in support of both.
//!
//! # Scope of this module
//!
//! - [`qfield`] — exact arithmetic in the field `Q(n)` and the polynomial /
//!   rational-function towers `Q(n)[k]`, `Q(n)(k)` built on top of it.
//! - [`hyperterm`] — recognition of *proper hypergeometric terms* `F(n, k)`
//!   (rational prefactor times `z^k w^n` times a product of
//!   `Γ(a·n + b·k + c)^e` factors with integer `a, b`) and their exact shift
//!   ratios `F(n,k+1)/F(n,k)` and `F(n+i,k)/F(n,k)`.
//! - [`mod@zeilberger`] — Zeilberger's creative-telescoping algorithm: given a
//!   proper hypergeometric `F(n, k)`, find a P-recursive relation
//!   `Σ_i a_i(n)·F(n+i,k) = G(n,k+1) − G(n,k)` with `G = R·F` and `R` an
//!   exact rational-function certificate. Deriving a recurrence for the *sum*
//!   `Σ_k F(n,k)` from it carries a boundary hypothesis that this module states
//!   rather than assumes — see [`zeilberger::boundary_side_condition`]. The
//!   order it returns is likewise not claimed to be minimal unless the search
//!   established it: see [`zeilberger::OrderSearch`] and
//!   [`zeilberger::ZeilbergerSearchReport::order_is_minimal`].
//! - [`boundary`] — *deciding* that boundary hypothesis over a stated summation
//!   range, three-valued: [`boundary::BoundaryStatus::Vanishes`] (the
//!   homogeneous recurrence holds for the sum),
//!   [`boundary::BoundaryStatus::Nonzero`] (the inhomogeneous one does, with the
//!   boundary term explicit) or [`boundary::BoundaryStatus::Unknown`] (nothing
//!   may be claimed about the sum).
//! - [`modular`] — evaluation of a P-recursive sequence *modulo `p^k`* directly
//!   from its recurrence, plus `binomial(a, b) mod p^k`. This is the evidence
//!   half of supercongruence work: reduce first and iterate in `ℤ/p^K`, rather
//!   than computing `S(N)` over `ℤ` and reducing a number with `Θ(N)` digits.
//!   Indices where the leading coefficient is not a unit mod `p` are handled by
//!   lifting to a higher working precision — never by dividing anyway.
//! - [`asymptotics`] — the natural *second* question after a certified
//!   recurrence: how fast does the sequence grow? Poincaré–Perron reads the
//!   growth rate `ρ` and the polynomial exponent `α` in `u(n) ~ C·ρⁿ·n^α`
//!   straight off the coefficient polynomials. The connection constant `C` does
//!   **not** follow from them — it depends on the initial conditions — so it is
//!   fitted from the terms and reported separately as
//!   [`asymptotics::ConnectionConstant`], never mixed in with the derived half.
//!   Equal-modulus roots, a repeated dominant root, a degenerate leading
//!   coefficient and an eventually-zero sequence each get their own
//!   [`asymptotics::PerronVerdict`] rather than a confident wrong number.
//!
//! Every certificate this module returns is checked as an *exact* identity
//! in `Q(n)(k)` before it is handed back to the caller — see
//! [`zeilberger::zeilberger()`] — so a successful call is a proof, not a
//! heuristic match.  When the search is inconclusive (order/degree bounds
//! exhausted) or the input is outside the supported class, the module
//! refuses via [`HolonomicError`] rather than guessing.
//!
//! Ore-operator closure (`ore.rs`) and ODE guessing from a power series are
//! tracked as follow-up work and are not part of this module yet; see
//! `ROADMAP.md` (P1 item 7). *Recurrence* guessing from finite data ships as
//! `alkahest.guess_holonomic` on the Python side, where the only mathematical
//! step is an exact nullspace the kernel already provides.

pub mod asymptotics;
pub mod boundary;
pub mod hyperterm;
pub mod modular;
pub mod qfield;
pub mod qzeil;
pub mod zeilberger;

pub use asymptotics::{
    asymptotics_from_recurrence, CharacteristicAnalysis, CharacteristicRoot, ConnectionConstant,
    PerronVerdict, RecurrenceAsymptotics,
};
pub use boundary::{boundary_status, natural_limits, BoundaryStatus};
pub use hyperterm::{GammaFactor, ProperTerm};
pub use modular::{binomial_mod, ModularEvaluation, ModularRecurrence};
pub use qfield::{PolyK, RatK, Rn};
pub use qzeil::{
    q_boundary_status, q_zeilberger, QBoundaryStatus, QCertificate, QHolonomicError, QProperTerm,
    QZeilbergerOpts, QZeilbergerReport, QZeilbergerResult,
};
pub use zeilberger::{
    boundary_side_condition, boundary_term, zeilberger, zeilberger_search, OrderSearch,
    ZeilbergerOpts, ZeilbergerResult, ZeilbergerSearchReport,
};

use std::fmt;

/// Errors from the holonomic subsystem (proper-hypergeometric recognition
/// and Zeilberger's algorithm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HolonomicError {
    /// The input expression does not fit the proper hypergeometric class
    /// this module supports (rational prefactor times `z^k w^n` times
    /// `Γ(a·n + b·k + c)^e` factors with `a, b ∈ ℤ`).
    NotProperHypergeometric(String),
    /// The bounded search (order and/or certificate degree) was exhausted
    /// without finding a certificate that passed exact verification.
    SearchExhausted(String),
    /// A candidate certificate was found by the search but failed the exact
    /// `Q(n)(k)` identity check. This should not happen for a correct
    /// implementation on a genuinely proper hypergeometric input; it is
    /// refused rather than returned unverified.
    CertificateVerificationFailed(String),
    /// Malformed call (e.g. `n` and `k` not distinct, non-positive bounds).
    InvalidInput(String),
    /// The modulus is not a prime power this subsystem can work over: the base
    /// is composite, the exponent is zero, or `p^k` is past the machine-word
    /// backend's ceiling. See [`modular`].
    ModulusUnsupported(String),
    /// A step of the recurrence does not determine the next term as a `p`-adic
    /// integer: the leading coefficient vanishes identically at that index, or
    /// the numerator's `p`-adic valuation is below the leading coefficient's.
    /// See [`modular`] for how singular indices are handled when they *are*
    /// determined.
    PAdicallyUndetermined(String),
    /// The computation is well posed but past a resource budget — a working
    /// precision the machine-word modulus cannot hold, or a `binomial_mod`
    /// whose cost is dominated by a pass over `1 … p−1`.
    WorkLimitExceeded(String),
}

impl fmt::Display for HolonomicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HolonomicError::NotProperHypergeometric(s) => {
                write!(f, "holonomic: not a proper hypergeometric term: {s}")
            }
            HolonomicError::SearchExhausted(s) => {
                write!(f, "holonomic: search exhausted: {s}")
            }
            HolonomicError::CertificateVerificationFailed(s) => {
                write!(f, "holonomic: certificate failed exact verification: {s}")
            }
            HolonomicError::InvalidInput(s) => {
                write!(f, "holonomic: invalid input: {s}")
            }
            HolonomicError::ModulusUnsupported(s) => {
                write!(f, "holonomic: unsupported modulus: {s}")
            }
            HolonomicError::PAdicallyUndetermined(s) => {
                write!(f, "holonomic: not determined p-adically: {s}")
            }
            HolonomicError::WorkLimitExceeded(s) => {
                write!(f, "holonomic: work limit exceeded: {s}")
            }
        }
    }
}

impl std::error::Error for HolonomicError {}

impl crate::errors::AlkahestError for HolonomicError {
    fn code(&self) -> &'static str {
        match self {
            HolonomicError::NotProperHypergeometric(_) => "E-HOLO-001",
            HolonomicError::SearchExhausted(_) => "E-HOLO-002",
            HolonomicError::CertificateVerificationFailed(_) => "E-HOLO-003",
            HolonomicError::InvalidInput(_) => "E-HOLO-004",
            HolonomicError::ModulusUnsupported(_) => "E-HOLO-006",
            HolonomicError::PAdicallyUndetermined(_) => "E-HOLO-007",
            HolonomicError::WorkLimitExceeded(_) => "E-HOLO-008",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            HolonomicError::NotProperHypergeometric(_) => {
                "rewrite the term as R(n,k)*z**k*w**n*prod(gamma(a*n + b*k + c)**e) with \
                 integer a, b and rational c; supported function heads are gamma, factorial, \
                 binomial, pochhammer"
            }
            HolonomicError::SearchExhausted(_) => {
                "raise max_order and/or max_degree in ZeilbergerOpts; if the term genuinely \
                 has no such recurrence within reach, Zeilberger's algorithm does not apply"
            }
            HolonomicError::CertificateVerificationFailed(_) => {
                "internal: report the term as a minimal failing example"
            }
            HolonomicError::InvalidInput(_) => {
                "n and k must be distinct symbols; max_order and max_degree must be positive"
            }
            HolonomicError::ModulusUnsupported(_) => {
                "the modulus must be p**k with p prime, k >= 1 and p**k < 2**62; for a \
                 composite modulus, evaluate at each prime power and recombine by CRT"
            }
            HolonomicError::PAdicallyUndetermined(_) => {
                "no modulus repairs this: the recurrence itself leaves Z_p at that index. \
                 Supply more initial terms so the evaluation starts past it, use a \
                 recurrence whose leading coefficient does not vanish there, or accept \
                 that the sequence is not p-integral and rescale it"
            }
            HolonomicError::WorkLimitExceeded(_) => {
                "lower k, use a smaller prime, or ask for an index the recurrence reaches \
                 without crossing so many singular steps"
            }
        })
    }
}
