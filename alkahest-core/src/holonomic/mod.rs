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
//!   exact rational-function certificate.
//!
//! Every certificate this module returns is checked as an *exact* identity
//! in `Q(n)(k)` before it is handed back to the caller — see
//! [`zeilberger::zeilberger()`] — so a successful call is a proof, not a
//! heuristic match.  When the search is inconclusive (order/degree bounds
//! exhausted) or the input is outside the supported class, the module
//! refuses via [`HolonomicError`] rather than guessing.
//!
//! Ore-operator closure (`ore.rs`) and recurrence/ODE guessing from finite
//! data (`guess.rs`) are tracked as follow-up work and are not part of this
//! module yet; see `ROADMAP.md` (P1 item 7).

pub mod hyperterm;
pub mod qfield;
pub mod zeilberger;

pub use hyperterm::{GammaFactor, ProperTerm};
pub use qfield::{PolyK, RatK, Rn};
pub use zeilberger::{zeilberger, ZeilbergerOpts, ZeilbergerResult};

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
        })
    }
}
