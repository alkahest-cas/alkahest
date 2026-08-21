//! `q`-analogue creative telescoping: `q`-Zeilberger, with a boundary verdict.
//!
//! This is the `q`-branch of M4. It proves recurrences for `q`-hypergeometric
//! sums — `q`-binomial (Gaussian) coefficients, `q`-Pochhammer symbols,
//! `q`-Vandermonde and its relatives — which the classical
//! [`mod@super::zeilberger`] cannot express at all, because none of those terms is
//! a proper hypergeometric term in `(n, k)`.
//!
//! ```text
//! Σ_{i=0}^{J} a_i(qⁿ)·F(n+i, k) = G(n, k+1) − G(n, k),   G = R·F
//! ```
//!
//! with `a_i ∈ Q(q)[qⁿ]` and `R ∈ Q(q)(qⁿ)(q^k)`, **re-checked as an exact
//! identity in `Q(q)(qⁿ)(q^k)` before it is returned** — the same
//! non-negotiable discipline as the classical module's: a returned certificate
//! is a proof, not a match.
//!
//! # What is supported, exactly
//!
//! The class is [`term::QProperTerm`]'s and it is enforced by the parser:
//!
//! ```text
//! F(n, k) = R(qⁿ, q^k) · z^k · w^n · q^{A·k² + B·n·k + C·n² + D·k + E·n}
//!           · ∏_j (q^{a_j·n + b_j·k + c_j}; q^{d_j})_{p_j·n + r_j·k + s_j}^{e_j}
//! ```
//!
//! Written as an expression, that is: `qbinomial(N, K)` and
//! `qpochhammer(u, d, v)` heads, powers of `q` with a degree-≤2 exponent in
//! `n, k`, powers with a base in `Q(q)`, and any rational function of `q`, `qⁿ`
//! and `q^k`. Everything else — a bare `n` or `k` outside an exponent, a
//! `Γ`, a `sin`, a second `q`-like parameter — is refused with a coded error
//! ([`QHolonomicError`], `E-HOLO-020`…`E-HOLO-024`), never approximated.
//!
//! Two in-class-looking inputs are refused as [`QHolonomicError::Unsupported`]
//! rather than answered: a Pochhammer whose first argument shifts by something
//! its base `q^d` does not divide (the shift quotient is an infinite product,
//! e.g. `(q; q²)_k` under `k ↦ k+1`), and a quadratic exponent whose shift
//! quotient is not an integer power of `q`.
//!
//! # `q` is generic
//!
//! Everything here is exact arithmetic in `Q(q)` with `q` **transcendental**.
//! A verdict is an identity of rational functions of `q`; it does *not* license
//! specialising `q` to a root of unity, which is exactly what the
//! `q`-supercongruence literature does. Specialisation is a separate step with
//! its own hypotheses, and this module does not take it.
//!
//! # The boundary question, in the `q` world
//!
//! The certificate is an identity about the *summand*. A recurrence for
//! `S(n) = Σ_k F(n,k)` is a second statement, and — as PR #303 established for
//! the classical case — assuming it is how a valid certificate becomes a false
//! theorem. [`q_boundary_status`] decides it, and it is **two-valued** here:
//! [`QBoundaryStatus::Vanishes`] (proved) or [`QBoundaryStatus::Unknown`]
//! (nothing about the sum may be claimed). There is deliberately no `Nonzero`
//! arm: computing the inhomogeneity `b(n)` exactly needs endpoint values of `G`
//! that are not rational in `qⁿ`, and returning an unproved `b(n)` would be
//! worse than returning nothing.
//!
//! ## Why `Vanishes` is a proof
//!
//! Read `G(n, ·)` the way [`super::boundary`] reads its own: as the *meromorphic
//! continuation* in `k`, not as the naive product of two values. The two differ,
//! and where they differ is the whole difficulty — the certificate really does
//! have poles at integer `k`. On `Σ_k [n;k]_q²·q^{k²}` the returned `R` has a
//! double pole at `q^k = q^{n+1}`, exactly where the summand has a double zero,
//! and `G(n, n+1)` is a finite **non-zero** limit of `0·∞`. A proof that
//! evaluated `R·F` factor-wise there would be wrong.
//!
//! Nothing below evaluates it. Fix `q` with `0 < |q| < 1` and any `n ≥ n_min`:
//!
//! 1. **Support.** [`term::QProperTerm::support`] proves, structurally, that
//!    `F(n+i, k) = 0` for every integer `k` outside an affine window, and that
//!    `F(n+i, k)` is *finite* at every integer `k`. A `q`-Pochhammer is exactly
//!    zero when one of its factors is `1 − q⁰` and exactly infinite when the
//!    same happens inside the reciprocal product a negative length denotes;
//!    both are linear conditions on `(n,k)` plus a divisibility, decided by
//!    Fourier–Motzkin over the rationals — which is complete, so a region
//!    proved empty is empty over the integers too. So the left-hand side
//!    `L(k) = Σ_i a_i(qⁿ)·F(n+i,k)` is finite at *every* integer `k` and zero
//!    outside a finite window (the `a_i` are polynomials in `qⁿ`, hence finite).
//! 2. **`G` is `0` far out on the right.** `R` is a rational function of `q^k`,
//!    so it has finitely many poles; at an integer `k` beyond both the window
//!    and those poles, `G(n,k) = R·0 = 0` with no indeterminacy.
//! 3. **Finiteness propagates from there.** `G(n,k) = G(n,k+1) − L(k)` with
//!    `L(k)` finite, so downward induction from step 2 makes *every* `G(n,k)`
//!    finite — including at the poles, where this is the only argument that
//!    gives the limit a value. Upward induction does the same to the right.
//! 4. **`G` vanishes at both ends.** Beyond the window `L ≡ 0`, so `G` is
//!    constant there; step 2 makes it `0` at infinitely many of those `k`, so it
//!    is `0` at all of them, poles included. Same at `−∞`.
//! 5. Summing the identity over `k ∈ Z` telescopes to `G(+∞) − G(−∞) = 0`, and
//!    the left-hand side is `Σ_i a_i(qⁿ)·S(n+i)` **with no moving-limit
//!    correction**, because the range does not move with `n`.
//!
//! Step 5 is what the fixed range buys. The classical module sums over
//! `k = 0..n`, whose limits move with `n`, and pays for it with the `D_i`
//! correction terms in [`super::boundary`] *and* with order counting at the
//! endpoints. Over `Z` there are neither: the poles are handled by the induction
//! in step 3 rather than by evaluating anything at them.
//!
//! The conclusion is then a statement about a rational function of `q` — both
//! `S(n)` and the `a_i` are — that holds on an open set of `q`, so it holds
//! identically in `Q(q)`.
//!
//! Two residual hypotheses, stated rather than hidden in
//! [`QBoundaryStatus::side_conditions`]: the verdict is about integers
//! `n ≥ n_min` at which the coefficients `a_i(qⁿ)` are defined, and `q` is
//! generic (see above).

pub mod cyclotomic;
pub mod field;
pub mod rootofunity;
pub mod search;
pub mod term;

pub use cyclotomic::{cyclotomic_polynomial, CycloElem, CycloField, MAX_CYCLOTOMIC_ORDER};
pub use rootofunity::{
    q_specialize_at_root_of_unity, QRootOfUnitySpecialization, QRootOfUnityStatus,
};
pub use search::{q_zeilberger_on_term, QZeilbergerOpts, QZeilbergerReport, QZeilbergerResult};
pub use term::{QProperTerm, QSupport};

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::holonomic::qfield::{rn_add, rn_is_zero, rn_mul, rn_one, Rn};
use crate::kernel::{ExprId, ExprPool};
use rug::Rational;
use std::fmt;

/// Errors from the `q`-analogue half of the holonomic subsystem.
///
/// Codes are `E-HOLO-020`…`E-HOLO-024`, disjoint from the classical
/// `E-HOLO-001`…`E-HOLO-005`, so a caller can tell which engine refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QHolonomicError {
    /// The input is not a `q`-proper hypergeometric term.
    NotQHypergeometric(String),
    /// The bounded `(order, degree)` search was exhausted.
    SearchExhausted(String),
    /// A candidate failed the exact `Q(q)(qⁿ)(q^k)` identity check. Refused
    /// rather than returned unverified.
    CertificateVerificationFailed(String),
    /// Malformed call (coincident symbols, non-positive bounds, a base step
    /// below 1).
    InvalidInput(String),
    /// In the shape of the class but outside the part of it this module can
    /// handle exactly — a Pochhammer shift the base does not divide, a
    /// quadratic exponent with a non-integral shift quotient, a span past the
    /// implementation limits.
    Unsupported(String),
}

impl fmt::Display for QHolonomicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QHolonomicError::NotQHypergeometric(s) => {
                write!(f, "q-holonomic: not a q-hypergeometric term: {s}")
            }
            QHolonomicError::SearchExhausted(s) => write!(f, "q-holonomic: search exhausted: {s}"),
            QHolonomicError::CertificateVerificationFailed(s) => {
                write!(f, "q-holonomic: certificate failed exact verification: {s}")
            }
            QHolonomicError::InvalidInput(s) => write!(f, "q-holonomic: invalid input: {s}"),
            QHolonomicError::Unsupported(s) => write!(f, "q-holonomic: unsupported: {s}"),
        }
    }
}

impl std::error::Error for QHolonomicError {}

impl crate::errors::AlkahestError for QHolonomicError {
    fn code(&self) -> &'static str {
        match self {
            QHolonomicError::NotQHypergeometric(_) => "E-HOLO-020",
            QHolonomicError::SearchExhausted(_) => "E-HOLO-021",
            QHolonomicError::CertificateVerificationFailed(_) => "E-HOLO-022",
            QHolonomicError::InvalidInput(_) => "E-HOLO-023",
            QHolonomicError::Unsupported(_) => "E-HOLO-024",
        }
    }

    fn remediation(&self) -> Option<&'static str> {
        Some(match self {
            QHolonomicError::NotQHypergeometric(_) => {
                "write the summand with qbinomial(N, K), qpochhammer(u, d, v), powers of q with a \
                 degree-2 exponent in n and k, and rational functions of q, q**n and q**k; a bare \
                 n or k outside an exponent is not q-hypergeometric"
            }
            QHolonomicError::SearchExhausted(_) => {
                "raise max_order and/or max_degree; if the sum genuinely satisfies no such \
                 q-recurrence, q-Zeilberger does not apply"
            }
            QHolonomicError::CertificateVerificationFailed(_) => {
                "internal: report the term as a minimal failing example"
            }
            QHolonomicError::InvalidInput(_) => {
                "q, n and k must be three distinct symbols; max_order and max_degree must be at \
                 least 1; a q-Pochhammer base step must be at least 1"
            }
            QHolonomicError::Unsupported(_) => {
                "the term is q-hypergeometric in shape but its shift quotient is not a rational \
                 function of q**n and q**k — e.g. (q; q**2)_k shifted in k. No algorithm in this \
                 family applies; close the branch"
            }
        })
    }
}

/// The verdict on whether the certificate's recurrence holds for the **sum**.
///
/// Two-valued on purpose — see the [module documentation](self). `Vanishes` is
/// a proof; `Unknown` licenses nothing about the sum, and the certificate
/// remains a true statement about the summand alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QBoundaryStatus {
    /// Proved: `Σ_i a_i(qⁿ)·S(n+i) = 0` for `S(n) = Σ_{k ∈ Z} F(n,k)`, which
    /// the same analysis proves is a finite sum.
    Vanishes {
        /// The verdict is about integers `n ≥ n_min`.
        n_min: i64,
        /// The support window in `k`, when each side came out as a single
        /// affine bound: `F(n,k) = 0` for `k` outside it.
        support: Option<(String, String)>,
    },
    /// Not established. **Nothing** follows about the sum.
    Unknown {
        /// What stopped the proof.
        reason: String,
    },
}

impl QBoundaryStatus {
    /// `"vanishes"` or `"unknown"` — the stable tag to record.
    pub fn tag(&self) -> &'static str {
        match self {
            QBoundaryStatus::Vanishes { .. } => "vanishes",
            QBoundaryStatus::Unknown { .. } => "unknown",
        }
    }

    /// Whether a recurrence for the *sum* may be read off at all.
    pub fn implies_sum_recurrence(&self) -> bool {
        matches!(self, QBoundaryStatus::Vanishes { .. })
    }

    /// What is still assumed after this verdict, as plain strings.
    pub fn side_conditions(&self) -> Vec<String> {
        match self {
            QBoundaryStatus::Vanishes { n_min, support } => {
                let mut out = vec![
                    format!(
                        "the summand was proved to have finite support in k and to be finite at \
                         every integer k, so the homogeneous recurrence sum_i a_i(q**n)*S(n+i) = 0 \
                         holds for S(n) = sum over all integer k of F(n,k), for every integer \
                         n >= {n_min} at which the coefficients a_i(q**n) are defined"
                    ),
                    "q is treated as transcendental: this is an identity in Q(q) and does not by \
                     itself license specialising q to a root of unity"
                        .to_string(),
                ];
                if let Some((lo, hi)) = support {
                    out.push(format!(
                        "the sum over all integer k is the finite sum over {lo} <= k <= {hi}, \
                         where the summand was proved to vanish outside that window"
                    ));
                }
                out
            }
            QBoundaryStatus::Unknown { reason } => vec![
                format!(
                    "no recurrence for the sum follows from this certificate: {reason}. The \
                     verified statement is the telescoping identity in k for the summand, and \
                     nothing more"
                ),
                "q is treated as transcendental: every identity here is an identity in Q(q)"
                    .to_string(),
            ],
        }
    }
}

/// Decide the boundary hypothesis for `Σ_{k ∈ Z} F(n,k)` over `n ≥ n_min`.
///
/// See the [module documentation](self) for why the four structural facts this
/// checks add up to a proof, and why nothing is evaluated at an endpoint.
pub fn q_boundary_status(f: &QProperTerm, order: usize, n_min: i64) -> QBoundaryStatus {
    let mut support: Option<(String, String)> = None;
    for i in 0..=order as i64 {
        let s = f.support(i, n_min);
        if !s.finite {
            return QBoundaryStatus::Unknown {
                reason: format!(
                    "at the n-shift {i}, {}",
                    nonempty(
                        &s.reason,
                        "the summand could not be proved finite at every integer k"
                    )
                ),
            };
        }
        if !s.bounded_above || !s.bounded_below {
            return QBoundaryStatus::Unknown {
                reason: format!(
                    "at the n-shift {i}, {}",
                    nonempty(
                        &s.reason,
                        "the summand's support in k could not be bounded on both sides"
                    )
                ),
            };
        }
        if i == 0 {
            if let (Some(lo), Some(hi)) = (&s.lo, &s.hi) {
                support = Some((format_bound(lo), format_bound(hi)));
            }
        }
    }
    QBoundaryStatus::Vanishes { n_min, support }
}

fn nonempty<'a>(s: &'a str, fallback: &'a str) -> &'a str {
    if s.is_empty() {
        fallback
    } else {
        s
    }
}

fn format_bound(b: &term::Rational2) -> String {
    let (a, c) = (b.a.clone(), b.b.clone());
    if a == 0 {
        return format!("{c}");
    }
    let head = if a == 1 {
        "n".to_string()
    } else if a == -1 {
        "-n".to_string()
    } else {
        format!("{a}*n")
    };
    match c.cmp0() {
        std::cmp::Ordering::Equal => head,
        std::cmp::Ordering::Greater => format!("{head} + {c}"),
        std::cmp::Ordering::Less => format!("{head} - {}", -c),
    }
}

/// A verified `q`-Zeilberger certificate together with its boundary verdict.
#[derive(Debug, Clone)]
pub struct QCertificate {
    /// The verified certificate and recurrence.
    pub report: QZeilbergerReport,
    /// Whether the recurrence carries over to the sum, and over what range.
    pub boundary: QBoundaryStatus,
    /// The parsed summand, kept so a caller can evaluate exact `q`-series terms
    /// and check the recurrence independently — which is the only check that
    /// would have caught the classical A279013 failure.
    pub term: QProperTerm,
}

/// `q`-Zeilberger's algorithm: a verified `q`-recurrence for a
/// `q`-hypergeometric term `F(n, k)`, plus a verdict on the sum.
///
/// `q`, `n`, `k` must be three distinct symbols. Refuses with
/// [`QHolonomicError`] rather than guessing outside the supported class or
/// beyond the search bounds.
pub fn q_zeilberger(
    term: ExprId,
    q: ExprId,
    n: ExprId,
    k: ExprId,
    pool: &ExprPool,
    opts: &QZeilbergerOpts,
) -> Result<DerivedExpr<QCertificate>, QHolonomicError> {
    if q == n || q == k || n == k {
        return Err(QHolonomicError::InvalidInput(
            "q, the outer index n and the summation index k must be three distinct symbols".into(),
        ));
    }
    let f = QProperTerm::parse(term, q, n, k, pool)?;
    let report = q_zeilberger_on_term(&f, q, n, k, pool, opts)?;
    let boundary = q_boundary_status(&f, report.result.order, opts.n_min);

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(
        "q_zeilberger_certificate",
        term,
        report.result.certificate,
    ));
    Ok(DerivedExpr::with_log(
        QCertificate {
            report,
            boundary,
            term: f,
        },
        log,
    ))
}

// ---------------------------------------------------------------------------
// Exact q-series evaluation — the independent check
// ---------------------------------------------------------------------------

/// Largest number of explicit `1 − q^{…}` factors an evaluation may expand.
const MAX_EVAL_SPAN: i64 = 4096;

impl QProperTerm {
    /// `F(n₀, k₀)` as an exact element of `Q(q)`, or `None` where the term is
    /// infinite (a denominator `q`-Pochhammer that vanishes, a prefactor pole).
    ///
    /// Nothing here goes through the shift quotients the search uses, which is
    /// the point: a recurrence checked against these values is checked against
    /// the actual sequence, not against the machinery that produced it.
    pub fn value_at(&self, n0: i64, k0: i64) -> Option<Rn> {
        let mut acc = field::raty_at(&self.rat, n0, k0)?;
        acc = rn_mul(&acc, &qq_pow_of(&self.z, k0)?);
        acc = rn_mul(&acc, &qq_pow_of(&self.w, n0)?);
        // The quadratic exponent must be an integer at this point — it is for
        // every term in the class (`k(k−1)/2` and friends), but it is checked
        // rather than assumed.
        let (rn0, rk0) = (Rational::from(n0), Rational::from(k0));
        let e = self.quad.a_kk.clone() * rk0.clone() * rk0.clone()
            + self.quad.b_nk.clone() * rn0.clone() * rk0.clone()
            + self.quad.c_nn.clone() * rn0.clone() * rn0.clone()
            + self.quad.d_k.clone() * rk0.clone()
            + self.quad.e_n.clone() * rn0
            + self.quad.konst.clone();
        if *e.denom() != 1 {
            return None;
        }
        acc = rn_mul(&acc, &field::qq_pow(e.numer().to_i64()?));
        for f in &self.poch {
            let u = f.u.cn.checked_mul(n0)? + f.u.ck.checked_mul(k0)? + f.u.c0;
            let v = f.v.cn.checked_mul(n0)? + f.v.ck.checked_mul(k0)? + f.v.c0;
            match poch_value(u, f.d, v)? {
                PochValue::Finite(p) => {
                    if rn_is_zero(&p) {
                        if f.e > 0 {
                            return Some(crate::holonomic::qfield::rn_zero());
                        }
                        return None; // 1/0
                    }
                    acc = rn_mul(&acc, &qq_pow_of(&p, f.e as i64)?);
                }
                PochValue::Infinite => {
                    if f.e > 0 {
                        return None;
                    }
                    return Some(crate::holonomic::qfield::rn_zero());
                }
            }
        }
        Some(acc)
    }

    /// The proved support window in `k` at a concrete `n₀`: integers
    /// `lo ≤ k ≤ hi` outside which `F(n₀, k)` was **proved** to be exactly `0`.
    ///
    /// Refuses rather than guessing when the window was not established, or
    /// when it is wider than the evaluation limit.
    pub fn window_at(&self, n0: i64, n_min: i64) -> Result<(i64, i64), QHolonomicError> {
        let s = self.support(0, n_min);
        if !s.finite || !s.bounded_above || !s.bounded_below {
            return Err(QHolonomicError::Unsupported(format!(
                "the summand's support in k was not established, so its sum is not a finite sum \
                 this module can evaluate: {}",
                s.reason
            )));
        }
        let (Some(lo), Some(hi)) = (&s.lo, &s.hi) else {
            return Err(QHolonomicError::Unsupported(
                "the support window is not a single affine bound on each side".into(),
            ));
        };
        let lo_v = ceil_at(lo, n0);
        let hi_v = floor_at(hi, n0);
        if hi_v - lo_v > MAX_EVAL_SPAN {
            return Err(QHolonomicError::Unsupported(format!(
                "the support window at n = {n0} spans {} terms (limit {MAX_EVAL_SPAN})",
                hi_v - lo_v
            )));
        }
        Ok((lo_v, hi_v))
    }

    /// `S(n₀) = Σ_{k ∈ Z} F(n₀, k)` as an exact element of `Q(q)`.
    ///
    /// Uses the proved support window, so this is a finite sum whose value is a
    /// theorem about the whole `Z`-sum, not a truncation.
    pub fn sum_at(&self, n0: i64, n_min: i64) -> Result<Rn, QHolonomicError> {
        let (lo_v, hi_v) = self.window_at(n0, n_min)?;
        let mut acc = crate::holonomic::qfield::rn_zero();
        for k0 in lo_v..=hi_v {
            let v = self.value_at(n0, k0).ok_or_else(|| {
                QHolonomicError::Unsupported(format!(
                    "the summand is not finite at (n, k) = ({n0}, {k0})"
                ))
            })?;
            acc = rn_add(&acc, &v);
        }
        Ok(acc)
    }
}

fn ceil_at(b: &term::Rational2, n0: i64) -> i64 {
    let v = b.a.clone() * Rational::from(n0) + b.b.clone();
    let num = v.numer().clone();
    let den = v.denom().clone();
    let (q, r) = num.div_rem_floor(den);
    let mut out = q.to_i64().unwrap_or(0);
    if r != 0 {
        out += 1;
    }
    out
}

fn floor_at(b: &term::Rational2, n0: i64) -> i64 {
    let v = b.a.clone() * Rational::from(n0) + b.b.clone();
    let num = v.numer().clone();
    let den = v.denom().clone();
    num.div_rem_floor(den).0.to_i64().unwrap_or(0)
}

enum PochValue {
    Finite(Rn),
    Infinite,
}

/// `(q^u; q^d)_v` at integer `u`, `v` — exactly, including the `0` and `∞`
/// cases a factor `1 − q⁰` produces.
fn poch_value(u: i64, d: i64, v: i64) -> Option<PochValue> {
    if v == 0 {
        return Some(PochValue::Finite(rn_one()));
    }
    if v.abs() > MAX_EVAL_SPAN {
        return None;
    }
    let mut prod = rn_one();
    if v > 0 {
        for t in 0..v {
            let e = u.checked_add(d.checked_mul(t)?)?;
            if e == 0 {
                return Some(PochValue::Finite(crate::holonomic::qfield::rn_zero()));
            }
            prod = rn_mul(&prod, &one_minus_q_pow(e));
        }
        Some(PochValue::Finite(prod))
    } else {
        for t in 1..=(-v) {
            let e = u.checked_sub(d.checked_mul(t)?)?;
            if e == 0 {
                return Some(PochValue::Infinite);
            }
            prod = rn_mul(&prod, &one_minus_q_pow(e));
        }
        Some(PochValue::Finite(crate::holonomic::qfield::rn_inv(&prod)?))
    }
}

fn one_minus_q_pow(e: i64) -> Rn {
    crate::holonomic::qfield::rn_sub(&rn_one(), &field::qq_pow(e))
}

fn qq_pow_of(base: &Rn, e: i64) -> Option<Rn> {
    if e == 0 {
        return Some(rn_one());
    }
    if rn_is_zero(base) || e.unsigned_abs() > 4096 {
        return None;
    }
    let b = if e < 0 {
        crate::holonomic::qfield::rn_inv(base)?
    } else {
        base.clone()
    };
    let mut acc = rn_one();
    for _ in 0..e.unsigned_abs() {
        acc = rn_mul(&acc, &b);
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::field::RatX;
    use super::*;
    use crate::errors::AlkahestError;
    use crate::holonomic::qfield::{rn_eq, rn_inv, rn_sub, rn_zero};
    use crate::kernel::Domain;

    fn syms(pool: &ExprPool) -> (ExprId, ExprId, ExprId) {
        (
            pool.symbol("q", Domain::Real),
            pool.symbol("n", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    fn qbinom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        pool.func("qbinomial", vec![top, bot])
    }

    /// `Σ_k [2n; k]_q` — the summand issue #10 (2026-08-19) was logged for.
    ///
    /// Class-legal and cheap to *state*; before this module had any ceiling it
    /// ran for eight minutes at the documented defaults with no output and had
    /// to be killed, while its near-twin `Σ_k [n; k]_q` decides in half a
    /// second. The cost is fragile in the *input*, not in `max_order` /
    /// `max_degree`, which is why a ceiling on the shape of the search is not
    /// enough on its own.
    fn q_central_2n(pool: &ExprPool, n: ExprId, k: ExprId) -> ExprId {
        qbinom(pool, pool.mul(vec![pool.integer(2_i32), n]), k)
    }

    #[test]
    fn q_zeilberger_honours_a_wall_budget() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let term = q_central_2n(&pool, n, k);
        let opts = QZeilbergerOpts::default();
        let _guard = crate::budget::enter(
            crate::budget::Budget::new().with_wall(std::time::Duration::from_millis(300)),
        );
        let start = std::time::Instant::now();
        let err = q_zeilberger(term, q, n, k, &pool, &opts)
            .expect_err("a 300 ms budget cannot cover this search");
        // Loose by two orders of magnitude: the point is that the call
        // *returns*, where before it consulted no budget at all.
        assert!(start.elapsed().as_secs() < 60, "budget was not consulted");
        assert!(
            matches!(err, QHolonomicError::SearchExhausted(_)),
            "{err:?}"
        );
        let trip = crate::budget::take_trip().expect("the budget trip must be recorded");
        assert_eq!(trip.code(), "E-BUDGET-001");
    }

    #[test]
    fn q_zeilberger_refuses_at_a_resource_ceiling_rather_than_running_unbounded() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let term = q_central_2n(&pool, n, k);
        // The cheapest bounds the engine accepts above the trivial ones.
        let opts = QZeilbergerOpts {
            max_order: 2,
            max_degree: 2,
            ..QZeilbergerOpts::default()
        };
        let err = q_zeilberger(term, q, n, k, &pool, &opts)
            .expect_err("no q-recurrence of this shape is found for this summand");
        let QHolonomicError::SearchExhausted(msg) = &err else {
            panic!("expected SearchExhausted, got {err:?}");
        };
        assert!(
            msg.contains("resource ceilings"),
            "a ceiling refusal must say so — a caller that reads this as 'the grid was covered \
             and nothing exists' records a false negative. Got: {msg}"
        );
        // No budget was active, so this is the module's own ceiling, not a trip.
        assert_eq!(crate::budget::take_trip(), None);
    }

    /// `(q;q)_m` at an integer `m ≥ 0`, built straight from the definition —
    /// the independent yardstick the recurrence is checked against.
    fn q_poch_int(m: i64) -> Rn {
        let mut acc = rn_one();
        for t in 1..=m {
            acc = rn_mul(&acc, &rn_sub(&rn_one(), &field::qq_pow(t)));
        }
        acc
    }

    /// The Gaussian binomial `[N; K]_q`, from the definition.
    fn q_binom_int(nn: i64, kk: i64) -> Rn {
        if kk < 0 || kk > nn {
            return rn_zero();
        }
        let den = rn_mul(&q_poch_int(kk), &q_poch_int(nn - kk));
        rn_mul(&q_poch_int(nn), &rn_inv(&den).expect("nonzero"))
    }

    /// Σ_i a_i(q^{n₀})·S(n₀+i) must be exactly zero in `Q(q)`.
    fn assert_annihilates(cert: &QCertificate, upto: i64) {
        let order = cert.report.result.order as i64;
        let s: Vec<Rn> = (0..=(upto + order))
            .map(|m| cert.term.sum_at(m, 0).expect("the sum is finite"))
            .collect();
        for n0 in 0..=upto {
            let mut acc = rn_zero();
            for (i, a) in cert.report.result.coeffs_x.iter().enumerate() {
                let ai = field::polyx_at_qn(a, n0);
                acc = rn_add(&acc, &rn_mul(&ai, &s[(n0 + i as i64) as usize]));
            }
            assert!(
                rn_is_zero(&acc),
                "the recurrence must annihilate the exact q-series sum at n = {n0}"
            );
        }
    }

    /// **The flagship.** `Σ_k [n;k]_q²·q^{k²} = [2n;n]_q` — the `q`-Vandermonde
    /// convolution at `m = r = n`, and the `q`-analogue of `Σ_k C(n,k)² =
    /// C(2n,n)`.
    ///
    /// Verified three ways, on purpose:
    /// 1. the certificate is re-checked as an exact `Q(q)(qⁿ)(q^k)` identity
    ///    inside the search (it is not returned otherwise);
    /// 2. the boundary verdict proves the recurrence carries to the sum;
    /// 3. the recurrence is checked against the **actual** `q`-series terms,
    ///    computed from the definition of the `q`-Pochhammer symbol and never
    ///    through the shift quotients — which is the check that a valid
    ///    certificate implying a false sum recurrence (the classical A279013
    ///    failure) would not survive.
    #[test]
    fn q_vandermonde_square_sum() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let b = qbinom(&pool, n, k);
        let f = pool.mul(vec![b, b, pool.pow(q, pool.mul(vec![k, k]))]);

        let start = std::time::Instant::now();
        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("q-Zeilberger must decide the q-Vandermonde square sum")
            .value;
        println!(
            "q-Vandermonde: order {} in {:?} ({} probes)",
            cert.report.result.order,
            start.elapsed(),
            cert.report.probes
        );

        assert_eq!(cert.report.result.order, 1);
        assert_eq!(cert.boundary.tag(), "vanishes");
        assert!(cert.boundary.implies_sum_recurrence());
        println!(
            "  a_0 = {}\n  a_1 = {}\n  R   = {}",
            pool.display(cert.report.result.coeffs[0]),
            pool.display(cert.report.result.coeffs[1]),
            pool.display(cert.report.result.certificate)
        );

        // The recurrence is the known one: `(1 − q^{n+1})²·S(n+1) =
        // (1 − q^{2n+1})(1 − q^{2n+2})·S(n)`, which is what
        // `[2n;n]_q → [2n+2;n+1]_q` demands. Pinned as a ratio in `Q(q)(qⁿ)`,
        // so an overall scale does not matter.
        let x = field::ratx_x_pow(1);
        let qx = |e: i64, p: i64| {
            let mut acc = RatX::from_rn(field::qq_pow(e));
            for _ in 0..p {
                acc = acc.mul(&x);
            }
            RatX::one().sub(&acc)
        };
        let want = qx(1, 2)
            .mul(&qx(2, 2))
            .div(&qx(1, 1).mul(&qx(1, 1)))
            .expect("nonzero")
            .neg();
        let got = RatX::from_poly(cert.report.result.coeffs_x[0].clone())
            .div(&RatX::from_poly(cert.report.result.coeffs_x[1].clone()))
            .expect("the leading coefficient is nonzero");
        assert!(
            got.eq_ratk(&want),
            "expected a_0/a_1 = -(1-q*x^2)(1-q^2*x^2)/(1-q*x)^2 with x = q^n"
        );

        // The identity itself: S(n) = [2n; n]_q, in exact Q(q).
        for n0 in 0..6 {
            let s = cert.term.sum_at(n0, 0).expect("finite sum");
            assert!(
                rn_eq(&s, &q_binom_int(2 * n0, n0)),
                "sum_{{k}} [n;k]^2 q^{{k^2}} must be [2n;n]_q at n = {n0}"
            );
        }
        assert_annihilates(&cert, 5);
    }

    /// `Σ_k [n;k]_q` — the Galois numbers `G_n`, which satisfy the order-2
    /// recurrence `G_{n+1} = 2·G_n + (qⁿ − 1)·G_{n−1}`. A second identity, at a
    /// higher order, checked the same three ways.
    #[test]
    fn galois_numbers_order_two() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let f = qbinom(&pool, n, k);

        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("the Galois-number sum must be decided")
            .value;
        assert_eq!(cert.report.result.order, 2);
        assert_eq!(cert.boundary.tag(), "vanishes");

        // G_0..G_4 = 1, 2, 5, 16, 67 at q = 1; here they are q-polynomials, so
        // the check is against the definition rather than against integers.
        for n0 in 0..5 {
            let s = cert.term.sum_at(n0, 0).expect("finite sum");
            let mut want = rn_zero();
            for j in 0..=n0 {
                want = rn_add(&want, &q_binom_int(n0, j));
            }
            assert!(rn_eq(&s, &want), "G_{n0} must be the sum of its row");
        }
        assert_annihilates(&cert, 4);
    }

    /// The `q`-analogue of `Σ_k (−1)^k C(n,k) = 0`: the `q`-binomial theorem's
    /// alternating case `Σ_k (−1)^k q^{k(k−1)/2} [n;k]_q = 0` for `n ≥ 1`.
    ///
    /// `q^{k(k−1)/2}` is **not** a rational function of `q^k` — its exponent is
    /// half-integral — but every shift quotient of it is, which is exactly the
    /// case the class admits and the parser checks for.
    #[test]
    fn half_integral_quadratic_exponent_is_accepted() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let half = pool.rational(1, 2);
        let kk1 = pool.mul(vec![half, k, pool.add(vec![k, pool.integer(-1_i32)])]);
        let sign = pool.pow(pool.integer(-1_i32), k);
        let f = pool.mul(vec![sign, pool.pow(q, kk1), qbinom(&pool, n, k)]);

        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("the alternating q-binomial sum must be decided")
            .value;
        assert_eq!(cert.boundary.tag(), "vanishes");
        // The sum is 0 for every n ≥ 1, which the recurrence must respect.
        for n0 in 1..5 {
            let s = cert.term.sum_at(n0, 0).expect("finite sum");
            assert!(
                rn_is_zero(&s),
                "the alternating sum must vanish at n = {n0}"
            );
        }
        assert_annihilates(&cert, 4);
    }

    /// The certificate has a **pole** at `k = n+1`, where the summand has a
    /// double zero — the fact the boundary proof is built to survive.
    ///
    /// This is not a defect and it is not avoidable: `G(n, n+1)` is a finite
    /// non-zero limit of `0·∞`, so any argument that evaluated `R·F` factor-wise
    /// at the endpoint would be wrong there. The verdict is proved by inducting
    /// finiteness inwards from a `k` past every pole instead — see the module
    /// docs — and this test pins the premise so that a future "simplification"
    /// of that argument into an endpoint evaluation fails here.
    #[test]
    fn the_certificate_really_does_have_a_pole_at_the_boundary() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let b = qbinom(&pool, n, k);
        let f = pool.mul(vec![b, b, pool.pow(q, pool.mul(vec![k, k]))]);
        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("certificate")
            .value;
        let den = &cert.report.result.certificate_xy.den;
        for n0 in 1..5 {
            let at_boundary = field::polyy_at(den, n0, n0 + 1).expect("the denominator evaluates");
            assert!(
                rn_is_zero(&at_boundary),
                "R must be singular at k = n+1 (n = {n0}); if it is not, the summand's zero \
                 there is unmatched and the boundary argument is being tested against the \
                 wrong premise"
            );
            // …and the summand really is zero there, so the product is 0·∞.
            assert!(rn_is_zero(&cert.term.value_at(n0, n0 + 1).expect("finite")));
        }
    }

    /// Outside the class: refused with `E-HOLO-020`, not answered.
    #[test]
    fn refuses_non_q_hypergeometric_input() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let bad = pool.func("sin", vec![pool.mul(vec![n, k])]);
        let err = q_zeilberger(bad, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect_err("sin(nk) is not q-hypergeometric");
        assert!(matches!(err, QHolonomicError::NotQHypergeometric(_)));
        assert_eq!(err.code(), "E-HOLO-020");
    }

    /// A bare `k` outside an exponent is not `q`-hypergeometric either — the
    /// classical class and this one are genuinely different, and the parser
    /// does not quietly reinterpret one as the other.
    #[test]
    fn refuses_a_classical_hypergeometric_term() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let f = pool.mul(vec![
            qbinom(&pool, n, k),
            pool.pow(pool.add(vec![k, pool.integer(1_i32)]), pool.integer(-1_i32)),
        ]);
        let err = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect_err("1/(k+1) is not a rational function of q^k");
        assert_eq!(err.code(), "E-HOLO-020");
    }

    /// In the shape of the class but outside it: `(q; q²)_k` shifted in `k`
    /// moves its first argument by `1`, which the base `q²` does not divide, so
    /// the shift quotient is an infinite product. `E-HOLO-024`, not a guess.
    #[test]
    fn refuses_a_base_incompatible_shift() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        // (q^k; q^2)_n — the first argument moves by 1 under k ↦ k+1.
        let f = pool.func("qpochhammer", vec![k, pool.integer(2_i32), n]);
        let err = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect_err("a base-incompatible shift must be refused");
        assert!(matches!(err, QHolonomicError::Unsupported(_)));
        assert_eq!(err.code(), "E-HOLO-024");
    }

    /// Coincident symbols are a malformed call, not a silent reinterpretation.
    #[test]
    fn refuses_coincident_symbols() {
        let pool = ExprPool::new();
        let (q, n, _k) = syms(&pool);
        let err = q_zeilberger(n, q, n, n, &pool, &QZeilbergerOpts::default())
            .expect_err("n == k must be refused");
        assert_eq!(err.code(), "E-HOLO-023");
    }

    /// The honesty requirement: a summand whose support in `k` is *not*
    /// bounded below gets a certificate and **no** claim about its sum.
    ///
    /// `1/(q;q)_{n−k}` vanishes for `k > n` and is nonzero for every `k ≤ n`,
    /// so the telescoping identity is fine and the `Z`-sum does not exist. The
    /// verdict must be `"unknown"` — this is the case that would have produced
    /// a false theorem if the boundary were assumed.
    #[test]
    fn unbounded_support_yields_no_claim_about_the_sum() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let f = pool.pow(
            pool.func(
                "qpochhammer",
                vec![
                    pool.integer(1_i32),
                    pool.integer(1_i32),
                    pool.add(vec![n, pool.mul(vec![k, pool.integer(-1_i32)])]),
                ],
            ),
            pool.integer(-1_i32),
        );
        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("the telescoping identity itself is fine")
            .value;
        assert_eq!(cert.boundary.tag(), "unknown");
        assert!(!cert.boundary.implies_sum_recurrence());
        assert!(cert
            .boundary
            .side_conditions()
            .iter()
            .any(|s| s.contains("no recurrence for the sum follows")));
    }

    /// The support analysis is what the verdict rests on, so it is asserted
    /// directly: `[n;k]_q²·q^{k²}` is supported exactly on `0 ≤ k ≤ n`.
    #[test]
    fn support_of_the_q_binomial_square_is_zero_to_n() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let b = qbinom(&pool, n, k);
        let f = pool.mul(vec![b, b, pool.pow(q, pool.mul(vec![k, k]))]);
        let term = QProperTerm::parse(f, q, n, k, &pool).expect("in class");
        let s = term.support(0, 0);
        assert!(s.finite && s.bounded_above && s.bounded_below);
        let lo = s.lo.expect("a lower bound");
        let hi = s.hi.expect("an upper bound");
        assert_eq!(
            (lo.a.clone(), lo.b.clone()),
            (Rational::new(), Rational::new())
        );
        assert_eq!(
            (hi.a.clone(), hi.b.clone()),
            (Rational::from(1), Rational::new())
        );
        // …and the term really is zero just outside that window.
        for n0 in 0..4 {
            assert!(rn_is_zero(&term.value_at(n0, -1).expect("finite")));
            assert!(rn_is_zero(&term.value_at(n0, n0 + 1).expect("finite")));
        }
    }
}
