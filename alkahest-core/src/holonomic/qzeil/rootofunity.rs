//! Specialising a proved `Q(q)` identity at a root of unity `ζ_d`.
//!
//! [`super`] proves recurrences with `q` **transcendental**, and says so in
//! every verdict: an identity in `Q(q)` does not by itself license setting
//! `q = ζ_d`. This module takes that step, and it takes it as a *decision* with
//! three outcomes rather than as an assumption.
//!
//! # What is proved
//!
//! Fix a certificate whose boundary verdict is
//! [`QBoundaryStatus::Vanishes`]. Then
//!
//! ```text
//! Σ_{i=0}^{J} a_i(qⁿ)·S(n+i) = 0        in Q(q), for every integer n ≥ n_min
//! ```
//!
//! with `S(n) = Σ_{k ∈ Z} F(n,k)`, a **finite** sum over the proved support
//! window. Both `S(n)` and `a_i(qⁿ)` are concrete elements of `Q(q)`, so the
//! question "does this survive `q = ζ_d`?" is the question of whether the
//! specialisation homomorphism
//!
//! ```text
//! ev_ζ : Z_(Φ_d) → Q(ζ_d),      Z_(Φ_d) = { r ∈ Q(q) : v_{Φ_d}(r) ≥ 0 }
//! ```
//!
//! is *defined* on each of them — `Z_(Φ_d)` is the localisation of `Q[q]` at
//! the prime `(Φ_d)`, and `ev_ζ` is a ring homomorphism on it. That is a
//! divisibility question over `Q`, decided exactly in
//! [`super::cyclotomic`]: `r` has a value at `ζ_d` iff `Φ_d` does not divide
//! the denominator of `r` in lowest terms. Nothing is evaluated numerically.
//!
//! So the theorem this module discharges is:
//!
//! > If `v_{Φ_d}(a_i(q^{n₀})) ≥ 0` and `v_{Φ_d}(S(n₀+i)) ≥ 0` for every
//! > `i = 0…J`, then `Σ_i a_i(ζ_d^{n₀})·S_ζ(n₀+i) = 0` in `Q(ζ_d)`, where
//! > `S_ζ(m) = ev_ζ(S(m))`.
//!
//! Both hypotheses are checked, the generic identity is re-checked in `Q(q)` at
//! this `n₀`, and the specialised identity is re-checked in `Q(ζ_d)` before
//! anything is returned. If a hypothesis fails, the verdict is
//! [`QRootOfUnityStatus::Obstructed`] and **no** specialised identity is
//! offered.
//!
//! # Three things that go wrong here, and are reported rather than hidden
//!
//! 1. **A pole.** `S(m)` or a coefficient can have `v_{Φ_d} < 0`. Specialising
//!    anyway is the failure mode this module exists to prevent, and it is the
//!    `q`-analogue of the A279013 mistake: a certificate that re-checks
//!    perfectly while the specialised claim is false. Verdict:
//!    [`Obstructed`](QRootOfUnityStatus::Obstructed).
//!
//! 2. **Degeneracy.** The recurrence coefficients are rational in `q`, and a
//!    root of unity can kill them. For `Σ_k [n;k]_q²q^{k²}` the leading
//!    coefficient carries a factor of `1 + q`, so at `ζ_2` and `n = 1` the
//!    "recurrence" collapses to the single constraint `a_0·S_ζ(1) = 0` and no
//!    longer determines the next value; at `ζ_1` (the classical `q → 1` limit)
//!    *every* coefficient dies and the statement is `0 = 0`. Both are true
//!    statements and neither is a recurrence, so
//!    [`leading_coefficient_survives`] and [`is_vacuous`] report them rather
//!    than letting a caller iterate something that is not there.
//!
//! 3. **The window moves.** `[n;k]_q` at `ζ_d` obeys the `q`-Lucas theorem and
//!    vanishes at many `k` where it is non-zero generically — `[2;1]_q = 1 + q`
//!    is non-zero in `Q(q)` and zero at `ζ_2`. The support can therefore
//!    *shrink*, and [`effective_support`] records exactly where the surviving
//!    terms are. It can never *grow*: outside the generic window `F(m,k)` is
//!    the zero element of `Q(q)`, whose image under a ring homomorphism is `0`.
//!
//! # Two things this module does **not** do
//!
//! - It does not claim `S_ζ(m) = Σ_k ev_ζ(F(m,k))` unless every individual
//!   summand is also regular at `ζ_d`. When one is not, `S_ζ(m)` is still
//!   correct — it is the image of the exact `Q(q)` sum — but it is not the sum
//!   of the specialised summands, and
//!   [`is_termwise_regular`](QRootOfUnitySpecialization::is_termwise_regular)
//!   says so.
//! - It does not do creative microscoping. There is no free parameter `a` in
//!   the supported class to introduce and then send to `q^{-n}`; what is
//!   delivered is the specialisation step itself, together with the exact
//!   `Φ_d`-adic valuation of each `S(m)` — which is the quantity a
//!   `q`-supercongruence `Φ_d(q)^r | S(n)` asserts.
//!
//! [`effective_support`]: QRootOfUnitySpecialization::effective_support
//! [`leading_coefficient_survives`]: QRootOfUnitySpecialization::leading_coefficient_survives
//! [`is_vacuous`]: QRootOfUnitySpecialization::is_vacuous

use super::cyclotomic::{CycloElem, CycloField, MAX_CYCLOTOMIC_ORDER};
use super::field::polyx_at_qn;
use super::{QBoundaryStatus, QCertificate, QHolonomicError};
use crate::holonomic::qfield::{rn_add, rn_is_zero, rn_mul, rn_zero, Rn};

/// The verdict on whether a proved `Q(q)` recurrence survives `q = ζ_d`.
///
/// Three-valued on purpose, and the three are not interchangeable:
/// `Specializes` is a proof, `Obstructed` is a proof that the hypotheses fail
/// (a pole was *exhibited*, not merely suspected), and `Unknown` licenses
/// nothing at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRootOfUnityStatus {
    /// Proved: `Σ_i a_i(ζ_d^{n₀})·S_ζ(n₀+i) = 0` in `Q(ζ_d)`, re-checked in
    /// exact cyclotomic arithmetic before being returned.
    Specializes,
    /// A hypothesis was **proved** to fail: some `a_i(q^{n₀})` or `S(n₀+i)` has
    /// a pole at `ζ_d`. Nothing about the specialised sum is claimed — in
    /// particular this is not a proof that the sum itself is singular, only
    /// that this route to it is blocked.
    Obstructed {
        /// Which value, and what its `Φ_d`-adic valuation was.
        reason: String,
    },
    /// Not decided. **Nothing** follows.
    Unknown {
        /// What stopped the decision.
        reason: String,
    },
}

impl QRootOfUnityStatus {
    /// `"specializes"`, `"obstructed"` or `"unknown"` — the stable tag.
    pub fn tag(&self) -> &'static str {
        match self {
            QRootOfUnityStatus::Specializes => "specializes",
            QRootOfUnityStatus::Obstructed { .. } => "obstructed",
            QRootOfUnityStatus::Unknown { .. } => "unknown",
        }
    }

    /// Why the verdict came out as it did; empty for `Specializes`.
    pub fn reason(&self) -> &str {
        match self {
            QRootOfUnityStatus::Specializes => "",
            QRootOfUnityStatus::Obstructed { reason } | QRootOfUnityStatus::Unknown { reason } => {
                reason
            }
        }
    }
}

/// The result of specialising a `q`-Zeilberger certificate at `ζ_d`.
///
/// The `Q(q)`-side data (the exact valuations) is filled in whenever it could
/// be computed, *including* under `Obstructed` — a negative valuation is the
/// obstruction, and it is also the most interesting number here, since
/// `v_{Φ_d}(S(n)) ≥ r` is precisely the `q`-supercongruence `Φ_d(q)^r | S(n)`.
#[derive(Debug, Clone)]
pub struct QRootOfUnitySpecialization {
    /// The order of the root of unity.
    pub d: u32,
    /// The `n` the verdict is about; the recurrence relates `n₀ … n₀+J`.
    pub n0: i64,
    /// The verdict.
    pub status: QRootOfUnityStatus,
    /// `Φ_d(q)` and the arithmetic of `Q(ζ_d)`.
    pub field: CycloField,
    /// The proved generic support window `lo ≤ k ≤ hi` at `n₀`.
    pub window: Option<(i64, i64)>,
    /// `S_ζ(n₀+i)` for `i = 0…J`, when the specialisation went through.
    pub sums: Vec<CycloElem>,
    /// `v_{Φ_d}(S(n₀+i))`; a `None` entry means `S(n₀+i)` is identically zero,
    /// whose valuation is `+∞`.
    pub sum_valuations: Vec<Option<i64>>,
    /// `a_i(ζ_d^{n₀})` for `i = 0…J`, when the specialisation went through.
    pub coeffs: Vec<CycloElem>,
    /// The `k` inside the generic window at which `F_ζ(n₀,k) ≠ 0` — the
    /// *effective* window at `ζ_d`, which `q`-Lucas can make strictly smaller.
    pub effective_support: Vec<i64>,
    /// How many `k` in the generic window have `F(n₀,k) ≠ 0` in `Q(q)`.
    pub generic_support_size: usize,
    /// Whether every individual summand in the window is regular at `ζ_d`.
    pub termwise_regular: bool,
}

impl QRootOfUnitySpecialization {
    /// Whether a specialised recurrence may be claimed at all.
    pub fn specializes(&self) -> bool {
        matches!(self.status, QRootOfUnityStatus::Specializes)
    }

    /// Whether the specialised recurrence is `0 = 0`.
    ///
    /// True when every `a_i(ζ_d^{n₀})` vanishes. The statement is still a
    /// theorem; it is simply empty, and a caller that reads
    /// [`specializes`](Self::specializes) without reading this would be
    /// claiming more than it has.
    pub fn is_vacuous(&self) -> bool {
        self.specializes() && !self.coeffs.is_empty() && self.coeffs.iter().all(CycloElem::is_zero)
    }

    /// Whether the *leading* coefficient `a_J(ζ_d^{n₀})` survives, i.e. whether
    /// the specialised recurrence still determines `S_ζ(n₀+J)` from its
    /// predecessors.
    pub fn leading_coefficient_survives(&self) -> bool {
        self.specializes() && self.coeffs.last().is_some_and(|c| !c.is_zero())
    }

    /// Whether `S_ζ(m)` is also the sum of the specialised summands.
    ///
    /// `false` means at least one summand inside the window has a pole at
    /// `ζ_d`: [`sums`](Self::sums) is still the correct image of the exact
    /// `Q(q)` sum, but writing it as `Σ_k F_ζ(m,k)` would be writing down an
    /// undefined expression.
    pub fn is_termwise_regular(&self) -> bool {
        self.termwise_regular
    }

    /// Whether `q`-Lucas killed at least one term the generic identity needs.
    pub fn support_shrinks(&self) -> bool {
        self.effective_support.len() < self.generic_support_size
    }

    /// What is still assumed after this verdict, as plain strings.
    pub fn side_conditions(&self) -> Vec<String> {
        let d = self.d;
        let n0 = self.n0;
        match &self.status {
            QRootOfUnityStatus::Specializes => {
                let mut out = vec![format!(
                    "every coefficient a_i(q**{n0}) and every sum S({n0}+i) was proved to have \
                     non-negative Phi_{d}-adic valuation, so the specialisation map at a primitive \
                     {d}-th root of unity is defined on all of them and carries the proved Q(q) \
                     recurrence to sum_i a_i(zeta^{n0})*S_zeta({n0}+i) = 0 in Q(zeta_{d}); the \
                     specialised identity was re-checked in exact cyclotomic arithmetic"
                )];
                if self.is_vacuous() {
                    out.push(format!(
                        "the specialised recurrence is VACUOUS: every a_i(zeta^{n0}) vanishes at a \
                         primitive {d}-th root of unity, so the statement is 0 = 0 and constrains \
                         nothing. The specialised sum values are still correct"
                    ));
                } else if !self.leading_coefficient_survives() {
                    out.push(format!(
                        "the leading coefficient a_J(zeta^{n0}) vanishes at a primitive {d}-th \
                         root of unity, so the specialised recurrence does not determine the last \
                         value from the earlier ones"
                    ));
                }
                if !self.termwise_regular {
                    out.push(format!(
                        "at least one individual summand F(m, k) inside the window has a pole at a \
                         primitive {d}-th root of unity: the reported S_zeta values are the images \
                         of the exact Q(q) sums, and are NOT the sums of the specialised summands"
                    ));
                }
                if self.support_shrinks() {
                    out.push(format!(
                        "the support shrank under specialisation: {} of the {} terms that are \
                         non-zero in Q(q) vanish at a primitive {d}-th root of unity (the q-Lucas \
                         phenomenon). The sum is still over the same window; the vanishing terms \
                         simply contribute nothing",
                        self.generic_support_size - self.effective_support.len(),
                        self.generic_support_size
                    ));
                }
                out.push(
                    "zeta_d is any primitive d-th root of unity: the statement is an identity in \
                     Q(zeta_d) = Q[q]/(Phi_d(q)), so it holds for every primitive d-th root at \
                     once, not for one chosen embedding"
                        .to_string(),
                );
                out
            }
            QRootOfUnityStatus::Obstructed { reason } => vec![
                format!(
                    "the specialisation at a primitive {d}-th root of unity is obstructed: \
                     {reason}. Nothing about the specialised sum or recurrence follows; the proved \
                     statement remains the Q(q) one with q transcendental"
                ),
                "an obstruction here is a proof that this route is blocked, not a proof that the \
                 specialised identity is false"
                    .to_string(),
            ],
            QRootOfUnityStatus::Unknown { reason } => vec![format!(
                "the specialisation was not decided: {reason}. Nothing follows about q = zeta_{d}"
            )],
        }
    }
}

/// Specialise a verified `q`-Zeilberger certificate at a primitive `d`-th root
/// of unity, at the index `n₀`.
///
/// See the [module documentation](self) for the theorem and for the three ways
/// this can come out short of a usable statement. Returns
/// [`QHolonomicError::InvalidInput`] for a malformed request (`d = 0`, `n₀`
/// below the range the boundary verdict covers) and
/// [`QHolonomicError::Unsupported`] when a resource bound is hit; the
/// mathematical outcomes are all carried by
/// [`QRootOfUnityStatus`], never by an error.
///
/// `d = 1` is allowed and means `ζ_1 = 1`: the classical `q → 1` limit, where
/// `Q(ζ_1) = Q` and Gaussian binomials become ordinary ones.
pub fn q_specialize_at_root_of_unity(
    cert: &QCertificate,
    d: u32,
    n0: i64,
) -> Result<QRootOfUnitySpecialization, QHolonomicError> {
    let field = CycloField::new(d).ok_or_else(|| {
        QHolonomicError::InvalidInput(format!(
            "the order of the root of unity must be between 1 and {MAX_CYCLOTOMIC_ORDER}, got {d}"
        ))
    })?;

    let n_min = match &cert.boundary {
        QBoundaryStatus::Vanishes { n_min, .. } => *n_min,
        QBoundaryStatus::Unknown { reason } => {
            return Ok(unknown(
                field,
                n0,
                format!(
                    "the generic boundary verdict is already \"unknown\", so there is no proved \
                     Q(q) recurrence for the sum to specialise: {reason}"
                ),
            ));
        }
    };
    if n0 < n_min {
        return Err(QHolonomicError::InvalidInput(format!(
            "the boundary verdict covers n >= {n_min}, so it cannot be specialised at n = {n0}"
        )));
    }

    let order = cert.report.result.order;
    let window = cert.term.window_at(n0, n_min)?;

    // ---- The Q(q) side: exact sums, exact coefficients, exact valuations. ----
    let mut generic_sums: Vec<Rn> = Vec::with_capacity(order + 1);
    for i in 0..=order as i64 {
        generic_sums.push(cert.term.sum_at(n0 + i, n_min)?);
    }
    let generic_coeffs: Vec<Rn> = cert
        .report
        .result
        .coeffs_x
        .iter()
        .map(|a| polyx_at_qn(a, n0))
        .collect();
    if generic_coeffs.len() != generic_sums.len() {
        return Err(QHolonomicError::CertificateVerificationFailed(format!(
            "the certificate reports order {order} but carries {} coefficients",
            generic_coeffs.len()
        )));
    }

    // The premise, re-checked at this n0 rather than assumed from the verdict.
    let mut acc = rn_zero();
    for (c, s) in generic_coeffs.iter().zip(generic_sums.iter()) {
        acc = rn_add(&acc, &rn_mul(c, s));
    }
    if !rn_is_zero(&acc) {
        return Err(QHolonomicError::CertificateVerificationFailed(format!(
            "the proved Q(q) recurrence does not annihilate the exact q-series sums at n = {n0}; \
             refusing to specialise a premise that does not hold"
        )));
    }

    let sum_valuations: Vec<Option<i64>> =
        generic_sums.iter().map(|s| field.valuation(s)).collect();

    // ---- The hypotheses of the specialisation theorem, decided exactly. ----
    let mut coeffs = Vec::with_capacity(generic_coeffs.len());
    for (i, c) in generic_coeffs.iter().enumerate() {
        match field.specialize(c) {
            Some(v) => coeffs.push(v),
            None => {
                let v = field.valuation(c).unwrap_or(0);
                return Ok(obstructed(
                    field,
                    n0,
                    d,
                    window,
                    sum_valuations,
                    format!(
                        "the recurrence coefficient a_{i}(q**{n0}) has Phi_{d}-adic valuation {v} \
                         (< 0), i.e. a pole at a primitive {d}-th root of unity"
                    ),
                ));
            }
        }
    }
    let mut sums = Vec::with_capacity(generic_sums.len());
    for (i, s) in generic_sums.iter().enumerate() {
        match field.specialize(s) {
            Some(v) => sums.push(v),
            None => {
                let v = field.valuation(s).unwrap_or(0);
                let m = n0 + i as i64;
                return Ok(obstructed(
                    field,
                    n0,
                    d,
                    window,
                    sum_valuations,
                    format!(
                        "the sum S({m}) has Phi_{d}-adic valuation {v} (< 0), i.e. a pole at a \
                         primitive {d}-th root of unity"
                    ),
                ));
            }
        }
    }

    // ---- Re-check the specialised identity in exact cyclotomic arithmetic. ----
    let mut acc = field.zero();
    for (c, s) in coeffs.iter().zip(sums.iter()) {
        acc = field.add(&acc, &field.mul(c, s));
    }
    if !acc.is_zero() {
        return Err(QHolonomicError::CertificateVerificationFailed(format!(
            "the specialised recurrence failed its own re-check in Q(zeta_{d}) at n = {n0}; \
             refusing to return it"
        )));
    }

    // ---- Termwise regularity, and where the surviving terms are. ----
    let mut termwise_regular = true;
    for i in 0..=order as i64 {
        let m = n0 + i;
        let (lo, hi) = cert.term.window_at(m, n_min)?;
        for k in lo..=hi {
            let Some(v) = cert.term.value_at(m, k) else {
                termwise_regular = false;
                break;
            };
            if field.specialize(&v).is_none() {
                termwise_regular = false;
                break;
            }
        }
        if !termwise_regular {
            break;
        }
    }

    let mut effective_support = Vec::new();
    let mut generic_support_size = 0_usize;
    for k in window.0..=window.1 {
        let Some(v) = cert.term.value_at(n0, k) else {
            continue;
        };
        if rn_is_zero(&v) {
            continue;
        }
        generic_support_size += 1;
        match field.specialize(&v) {
            Some(sv) if !sv.is_zero() => effective_support.push(k),
            _ => {}
        }
    }

    Ok(QRootOfUnitySpecialization {
        d,
        n0,
        status: QRootOfUnityStatus::Specializes,
        field,
        window: Some(window),
        sums,
        sum_valuations,
        coeffs,
        effective_support,
        generic_support_size,
        termwise_regular,
    })
}

fn unknown(field: CycloField, n0: i64, reason: String) -> QRootOfUnitySpecialization {
    QRootOfUnitySpecialization {
        d: field.order(),
        n0,
        status: QRootOfUnityStatus::Unknown { reason },
        field,
        window: None,
        sums: Vec::new(),
        sum_valuations: Vec::new(),
        coeffs: Vec::new(),
        effective_support: Vec::new(),
        generic_support_size: 0,
        termwise_regular: false,
    }
}

fn obstructed(
    field: CycloField,
    n0: i64,
    d: u32,
    window: (i64, i64),
    sum_valuations: Vec<Option<i64>>,
    reason: String,
) -> QRootOfUnitySpecialization {
    QRootOfUnitySpecialization {
        d,
        n0,
        status: QRootOfUnityStatus::Obstructed { reason },
        field,
        window: Some(window),
        sums: Vec::new(),
        sum_valuations,
        coeffs: Vec::new(),
        effective_support: Vec::new(),
        generic_support_size: 0,
        termwise_regular: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::holonomic::qzeil::cyclotomic::cyclotomic_polynomial;
    use crate::holonomic::qzeil::{q_zeilberger, QZeilbergerOpts};
    use crate::kernel::{Domain, ExprId, ExprPool};
    use rug::{Integer, Rational};

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

    /// `Σ_k [n;k]_q²·q^{k²}`, whose sum is `[2n;n]_q`.
    fn vandermonde_cert(pool: &ExprPool) -> (QCertificate, ExprId, ExprId, ExprId) {
        let (q, n, k) = syms(pool);
        let b = qbinom(pool, n, k);
        let f = pool.mul(vec![b, b, pool.pow(q, pool.mul(vec![k, k]))]);
        let cert = q_zeilberger(f, q, n, k, pool, &QZeilbergerOpts::default())
            .expect("the q-Vandermonde square sum must be decided")
            .value;
        (cert, q, n, k)
    }

    // -----------------------------------------------------------------
    // Independent yardsticks. None of these touch the q-Zeilberger
    // machinery, the shift quotients, or `QProperTerm::value_at`.
    // -----------------------------------------------------------------

    /// `[N;K]_{ζ_d}` by the **Pascal recurrence** `[N;K] = [N−1;K−1] +
    /// ζ^K·[N−1;K]`, built directly in `Q(ζ_d)`.
    ///
    /// Deliberately a different algorithm from everything under test: no
    /// `q`-Pochhammer quotients, no rational functions, no specialisation —
    /// cyclotomic arithmetic from the first line.
    fn gaussian_binomial_at_zeta(f: &CycloField, nn: i64, kk: i64) -> CycloElem {
        if kk < 0 || kk > nn {
            return f.zero();
        }
        // row[j] = [i; j]_ζ
        let mut row = vec![f.one()];
        for _i in 1..=nn {
            let mut next = vec![f.zero(); row.len() + 1];
            for (j, cell) in row.iter().enumerate() {
                // [i;j] contributes ζ^j·[i−1;j] to [i;j] and [i−1;j] to [i;j+1].
                next[j] = f.add(&next[j], &f.mul(cell, &f.zeta_pow(j as i64)));
                next[j + 1] = f.add(&next[j + 1], cell);
            }
            row = next;
        }
        row[kk as usize].clone()
    }

    /// `[N;K]_{ζ_d}` by the **`q`-Lucas theorem**:
    /// `[N;K]_ζ = C(⌊N/d⌋, ⌊K/d⌋)·[N mod d; K mod d]_ζ`.
    ///
    /// A closed form via base-`d` digits and an *integer* binomial coefficient,
    /// with the residual small binomial built from the **product formula**
    /// `∏_{i=1}^{k}(1 − ζ^{n−k+i})/(1 − ζ^i)` — every factor of which is
    /// invertible because `0 < i ≤ k < d`. Structurally unrelated to the Pascal
    /// recurrence above and to everything under test.
    fn gaussian_binomial_by_q_lucas(f: &CycloField, nn: i64, kk: i64) -> CycloElem {
        let d = f.order() as i64;
        if kk < 0 || kk > nn {
            return f.zero();
        }
        let (n1, n0) = (nn / d, nn % d);
        let (k1, k0) = (kk / d, kk % d);
        if k0 > n0 || k1 > n1 {
            return f.zero();
        }
        let mut c = Integer::from(1);
        for j in 0..k1 {
            c *= Integer::from(n1 - j);
            c /= Integer::from(j + 1);
        }
        let one = f.one();
        let mut small = one.clone();
        for i in 1..=k0 {
            let num = f.sub(&one, &f.zeta_pow(n0 - k0 + i));
            let den = f.sub(&one, &f.zeta_pow(i));
            let inv = f.inv(&den).expect("1 − zeta^i is a unit for 0 < i < d");
            small = f.mul(&small, &f.mul(&num, &inv));
        }
        f.mul(&small, &f.from_rational(Rational::from(c)))
    }

    /// `v_{Φ_d}([N;K]_q) = ⌊N/d⌋ − ⌊K/d⌋ − ⌊(N−K)/d⌋`, from counting the
    /// multiples of `d` among the `1 − q^i` factors. Integer floors only.
    fn gaussian_binomial_valuation(d: i64, nn: i64, kk: i64) -> i64 {
        nn.div_euclid(d) - kk.div_euclid(d) - (nn - kk).div_euclid(d)
    }

    #[test]
    fn the_yardsticks_agree_with_each_other() {
        // Pascal and q-Lucas are independent; if they disagree the flagship
        // test below is measuring against a broken ruler.
        for d in 1_u32..=6 {
            let f = CycloField::new(d).expect("in range");
            for nn in 0..12 {
                for kk in 0..=nn {
                    assert_eq!(
                        gaussian_binomial_at_zeta(&f, nn, kk),
                        gaussian_binomial_by_q_lucas(&f, nn, kk),
                        "Pascal and q-Lucas disagree on [{nn};{kk}]_zeta_{d}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // The flagship
    // -----------------------------------------------------------------

    /// **`Σ_{k} [n;k]_q²·q^{k²} = [2n;n]_q`, specialised at `ζ_d`.**
    ///
    /// Checked against three things the specialisation machinery has no part
    /// in:
    ///
    /// 1. the sum recomputed term by term in `Q(ζ_d)` with the Gaussian
    ///    binomials built from the **Pascal recurrence**;
    /// 2. the closed form `[2n;n]_{ζ_d}` predicted by the **`q`-Lucas
    ///    theorem** from the base-`d` digits of `2n` and `n`;
    /// 3. the exact `Φ_d`-adic valuation of the sum, against the integer
    ///    floor count `⌊2n/d⌋ − 2⌊n/d⌋` of `Φ_d` factors in `[2n;n]_q`.
    ///
    /// (3) is the `q`-supercongruence content: it is the exact statement
    /// `Φ_d(q)^v ∥ Σ_k [n;k]_q²q^{k²}`.
    #[test]
    fn q_vandermonde_square_sum_at_a_root_of_unity() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);

        for d in 1_u32..=6 {
            let f = CycloField::new(d).expect("in range");
            for n0 in 0..9_i64 {
                let spec = q_specialize_at_root_of_unity(&cert, d, n0)
                    .expect("a polynomial summand has no pole anywhere");
                assert_eq!(
                    spec.status.tag(),
                    "specializes",
                    "d = {d}, n = {n0}: {}",
                    spec.status.reason()
                );
                // Every summand is a polynomial in q, so the termwise
                // statement holds too.
                assert!(spec.is_termwise_regular());

                // 1. The sum, recomputed in Q(ζ_d) by the Pascal recurrence.
                let mut direct = f.zero();
                for k in 0..=n0 {
                    let b = gaussian_binomial_at_zeta(&f, n0, k);
                    let sq = f.mul(&b, &b);
                    direct = f.add(&direct, &f.mul(&sq, &f.zeta_pow(k * k)));
                }
                assert_eq!(
                    spec.sums[0], direct,
                    "d = {d}, n = {n0}: the specialised sum must equal the sum computed \
                     independently in Q(zeta_{d})"
                );

                // 2. The closed form, by q-Lucas.
                let lucas = gaussian_binomial_by_q_lucas(&f, 2 * n0, n0);
                assert_eq!(
                    spec.sums[0], lucas,
                    "d = {d}, n = {n0}: sum_k [n;k]^2 q^(k^2) at zeta_{d} must be the q-Lucas \
                     value of [2n;n]"
                );

                // 3. The Phi_d-adic valuation — the supercongruence statement.
                let want_v = gaussian_binomial_valuation(d as i64, 2 * n0, n0);
                assert_eq!(
                    spec.sum_valuations[0],
                    Some(want_v),
                    "d = {d}, n = {n0}: Phi_{d}(q)^{want_v} must divide [2n;n]_q exactly"
                );
                // …and the two agree on whether the specialised value is zero.
                assert_eq!(want_v > 0, spec.sums[0].is_zero());
            }
        }
    }

    /// The specialised **recurrence** is checked against the specialised
    /// **values**, at every `(d, n)` where it is not vacuous.
    #[test]
    fn the_specialised_recurrence_annihilates_the_independent_values() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        let mut nontrivial = 0;
        for d in 2_u32..=6 {
            let f = CycloField::new(d).expect("in range");
            for n0 in 0..8_i64 {
                let spec = q_specialize_at_root_of_unity(&cert, d, n0).expect("no pole");
                if !spec.specializes() || spec.is_vacuous() {
                    continue;
                }
                nontrivial += 1;
                // Independent values, from q-Lucas rather than from `spec`.
                let mut acc = f.zero();
                for (i, c) in spec.coeffs.iter().enumerate() {
                    let s = gaussian_binomial_by_q_lucas(&f, 2 * (n0 + i as i64), n0 + i as i64);
                    acc = f.add(&acc, &f.mul(c, &s));
                }
                assert!(
                    acc.is_zero(),
                    "d = {d}, n = {n0}: the specialised recurrence must annihilate the q-Lucas \
                     values, not merely the ones it computed itself"
                );
            }
        }
        assert!(nontrivial >= 10, "the check must not be vacuously empty");
    }

    // -----------------------------------------------------------------
    // The window genuinely changes
    // -----------------------------------------------------------------

    /// `q`-Lucas kills terms: `[2;1]_q = 1 + q` is non-zero in `Q(q)` and zero
    /// at `ζ_2`, so the effective window at `d = 2, n = 2` is `{0, 2}` and not
    /// `{0, 1, 2}`.
    ///
    /// This must be **reported**, not silently absorbed — and the sum over the
    /// unchanged window must still be right.
    #[test]
    fn the_support_shrinks_at_a_root_of_unity_and_says_so() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        let spec = q_specialize_at_root_of_unity(&cert, 2, 2).expect("no pole");

        assert_eq!(spec.status.tag(), "specializes");
        assert_eq!(spec.window, Some((0, 2)));
        assert_eq!(spec.generic_support_size, 3);
        assert_eq!(
            spec.effective_support,
            vec![0, 2],
            "the k = 1 term must die at zeta_2, since [2;1]_q = 1 + q"
        );
        assert!(spec.support_shrinks());
        assert!(spec
            .side_conditions()
            .iter()
            .any(|s| s.contains("the support shrank under specialisation")));

        // …and the value is still the right one: [4;2]_{ζ_2} = C(2,1)·[0;0] = 2.
        let f = CycloField::new(2).expect("in range");
        assert_eq!(spec.sums[0], f.from_rational(Rational::from(2)));
    }

    /// The support can never *grow*: outside the proved window the summand is
    /// the zero element of `Q(q)`, and a ring homomorphism sends `0` to `0`.
    #[test]
    fn the_support_never_grows() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        for d in 1_u32..=5 {
            let f = CycloField::new(d).expect("in range");
            for n0 in 0..6_i64 {
                for k in [-3_i64, -1, n0 + 1, n0 + 4] {
                    let v = cert.term.value_at(n0, k).expect("finite");
                    assert_eq!(
                        f.specialize(&v),
                        Some(f.zero()),
                        "d = {d}, n = {n0}, k = {k}: outside the window the specialisation must \
                         still be zero"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Degeneracy
    // -----------------------------------------------------------------

    /// At `d = 1` (i.e. `q → 1`) the leading coefficient is a multiple of
    /// `(1 − q^{n+1})²`, so **every** coefficient dies and the specialised
    /// recurrence is `0 = 0`.
    ///
    /// The verdict is still `"specializes"` — the statement is true — and
    /// `is_vacuous` is what stops a caller from mistaking it for content. The
    /// *values* remain correct: at `q = 1` the identity is
    /// `Σ_k C(n,k)² = C(2n,n)`.
    #[test]
    fn the_classical_limit_is_a_vacuous_recurrence_with_correct_values() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        let f = CycloField::new(1).expect("in range");
        for n0 in 0..8_i64 {
            let spec = q_specialize_at_root_of_unity(&cert, 1, n0).expect("no pole");
            assert_eq!(spec.status.tag(), "specializes");
            assert!(
                spec.is_vacuous(),
                "at q = 1 the coefficients (1 − q^(n+1))² and (1 − q^(2n+1))(1 − q^(2n+2)) all \
                 vanish, so the specialised recurrence says nothing"
            );
            assert!(!spec.leading_coefficient_survives());
            assert!(spec.side_conditions().iter().any(|s| s.contains("VACUOUS")));

            // C(2n, n), computed as an integer.
            let mut c = Integer::from(1);
            for j in 0..n0 {
                c *= Integer::from(2 * n0 - j);
                c /= Integer::from(j + 1);
            }
            assert_eq!(
                spec.sums[0],
                f.from_rational(Rational::from(c)),
                "at q = 1 the sum must be the central binomial coefficient C(2n, n)"
            );
        }
    }

    /// The partial degeneracy, at a genuine root of unity: at `d = 2, n = 1`
    /// the **leading** coefficient `a_1(ζ_2)` vanishes while `a_0(ζ_2) = 4`
    /// does not.
    ///
    /// The specialised recurrence is then not a recurrence at all — it is the
    /// single constraint `4·S_ζ(1) = 0`, which no longer determines `S_ζ(2)`
    /// from `S_ζ(1)`. A caller that iterated it forwards would divide by zero,
    /// so the verdict has to say this out loud even though it is a true
    /// statement (and it is: `[2;1]_{ζ_2} = 1 + q|_{q=−1} = 0`).
    #[test]
    fn a_root_of_unity_can_kill_the_leading_coefficient() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        let f = CycloField::new(2).expect("in range");
        let spec = q_specialize_at_root_of_unity(&cert, 2, 1).expect("no pole");

        assert_eq!(spec.status.tag(), "specializes");
        assert!(!spec.is_vacuous(), "a_0 survives, so this is not vacuous");
        assert!(
            !spec.leading_coefficient_survives(),
            "a_1(zeta_2^1) is a multiple of (1 − q^2)|_{{q=−1}} = 0"
        );
        assert!(spec
            .side_conditions()
            .iter()
            .any(|s| s.contains("the leading coefficient")));
        // The surviving constraint really does force the value it claims.
        assert!(!spec.coeffs[0].is_zero());
        assert_eq!(spec.sums[0], f.zero());
        assert_eq!(spec.sum_valuations[0], Some(1));
    }

    // -----------------------------------------------------------------
    // Refusals
    // -----------------------------------------------------------------

    /// A **pole** at `ζ_d`: `Σ_k [n;k]_q²q^{k²}/(q³; q³)_1 = [2n;n]_q/(1 − q³)`
    /// has `Φ_3`-adic valuation `−1` whenever `3 ∤ ... ` — concretely at every
    /// `n` with `n mod 3 ≠ 2`, and since two consecutive `n` cannot both be
    /// `≡ 2 (mod 3)`, at *every* `n₀` one of `S(n₀)`, `S(n₀+1)` is singular.
    ///
    /// This is the A279013 hazard transplanted: the certificate is perfectly
    /// valid and re-checks cleanly in `Q(q)`, and specialising it at `ζ_3`
    /// anyway would produce a confidently wrong statement. It must be refused.
    #[test]
    fn a_pole_at_the_root_of_unity_is_obstructed_not_specialised() {
        let pool = ExprPool::new();
        let (q, n, k) = syms(&pool);
        let b = qbinom(&pool, n, k);
        // (q³; q³)_1 = 1 − q³, as a constant q-Pochhammer factor.
        let pole = pool.pow(
            pool.func(
                "qpochhammer",
                vec![
                    pool.integer(3_i32),
                    pool.integer(3_i32),
                    pool.integer(1_i32),
                ],
            ),
            pool.integer(-1_i32),
        );
        let f = pool.mul(vec![b, b, pool.pow(q, pool.mul(vec![k, k])), pole]);
        let cert = q_zeilberger(f, q, n, k, &pool, &QZeilbergerOpts::default())
            .expect("the certificate itself is fine — the constant factor cancels out of it")
            .value;
        assert_eq!(cert.boundary.tag(), "vanishes");

        let mut obstructed = 0;
        for n0 in 0..7_i64 {
            let spec =
                q_specialize_at_root_of_unity(&cert, 3, n0).expect("a verdict, not an error");
            assert_eq!(
                spec.status.tag(),
                "obstructed",
                "n = {n0}: 1/(1 − q³) has a pole at zeta_3 and must not be specialised"
            );
            assert!(!spec.specializes());
            assert!(spec.sums.is_empty(), "no specialised value may be offered");
            assert!(spec.status.reason().contains("valuation"));
            assert!(spec
                .side_conditions()
                .iter()
                .any(|s| s.contains("obstructed")));
            obstructed += 1;
        }
        assert_eq!(obstructed, 7);

        // The same certificate at a d where the factor is a unit specialises
        // fine — the refusal is about zeta_3, not about the term.
        let spec = q_specialize_at_root_of_unity(&cert, 5, 2).expect("no pole");
        assert_eq!(spec.status.tag(), "specializes", "{}", spec.status.reason());
    }

    /// A certificate whose *generic* verdict is `"unknown"` cannot be
    /// specialised either: there is no proved `Q(q)` statement to carry over.
    #[test]
    fn an_unknown_generic_verdict_stays_unknown_at_a_root_of_unity() {
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
            .expect("the telescoping identity is fine")
            .value;
        assert_eq!(cert.boundary.tag(), "unknown");

        let spec = q_specialize_at_root_of_unity(&cert, 3, 2).expect("a verdict, not an error");
        assert_eq!(spec.status.tag(), "unknown");
        assert!(!spec.specializes());
        assert!(spec.sums.is_empty());
        assert!(spec.status.reason().contains("already"));
    }

    /// Malformed requests are coded errors, not verdicts.
    #[test]
    fn malformed_requests_are_refused() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);

        let err = q_specialize_at_root_of_unity(&cert, 0, 3).expect_err("d = 0 names nothing");
        assert_eq!(err.code(), "E-HOLO-023");
        let err = q_specialize_at_root_of_unity(&cert, 100_000, 3).expect_err("past the cap");
        assert_eq!(err.code(), "E-HOLO-023");
        let err = q_specialize_at_root_of_unity(&cert, 3, -1)
            .expect_err("below the range the verdict covers");
        assert_eq!(err.code(), "E-HOLO-023");
    }

    /// The modulus really is the cyclotomic polynomial the caller can check
    /// against by hand.
    #[test]
    fn the_modulus_is_exposed_for_independent_checking() {
        let pool = ExprPool::new();
        let (cert, _q, _n, _k) = vandermonde_cert(&pool);
        let spec = q_specialize_at_root_of_unity(&cert, 6, 3).expect("no pole");
        assert_eq!(spec.field.modulus(), &cyclotomic_polynomial(6));
        assert_eq!(spec.field.degree(), 2); // φ(6) = 2
    }
}
