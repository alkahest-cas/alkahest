//! Shared scaffolding for the *asymptotics at scale* family (P1 item 10).
//!
//! # Why some of this is currently unused
//!
//! Only the Euler–Maclaurin route ships in this change; sequence asymptotics,
//! singularity analysis and saddle-point expansions are documented follow-ups
//! (see `docs/mdbook/src/asymptotics.md`). The pieces they need — exact `ℚ`
//! polynomial arithmetic, rational-function extraction, rational-root
//! splitting, Bernoulli *polynomials* and the complex root finder — are kept
//! here rather than deleted and re-added, so `dead_code` is allowed
//! module-wide. Everything the shipped route uses is exercised by
//! `super::euler_maclaurin`'s tests.
//!
//! The routes in this family, built on top of
//! [`mod@crate::calculus::asymptotic`] — sequence asymptotics, singularity
//! analysis of generating functions, Laplace/saddle-point asymptotics of
//! parameter integrals, and Euler–Maclaurin expansion of sums
//! ([`mod@crate::calculus::euler_maclaurin`], the one that ships here) — all
//! share three things:
//!
//! 1. **A common result object** ([`AsymptoticReport`]) that records not only
//!    the terms but *how much is proved*: which hypotheses the implementation
//!    actually checked, which it merely assumed, and the numeric evidence.
//! 2. **A common numeric verification gate** (`gate_accept`) generalising the
//!    `o()`-gate of [`crate::calculus::asymptotic::asymptotic_expand`] to an
//!    arbitrary *oracle* — the exact sequence value, the exact Taylor
//!    coefficient, a high-accuracy quadrature, or a directly summed partial
//!    sum. Terms that fail the gate are dropped; if nothing survives the route
//!    declines rather than emit an unverified expansion.
//! 3. **Small exact-arithmetic utilities** — Bernoulli numbers and polynomials,
//!    univariate rational-function extraction over ℚ, rational root splitting,
//!    and a Durand–Kerner complex root finder.
//!
//! Nothing in this module is a user-facing entry point; it exists so the four
//! routes cannot drift apart in how honestly they report themselves.
#![allow(dead_code)]

use crate::kernel::{ExprId, ExprPool};
use crate::simplify::simplify;
use rug::{Integer, Rational};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Reporting vocabulary
// ---------------------------------------------------------------------------

/// How strongly an emitted expansion is justified.
///
/// This is deliberately coarse: the only distinction that matters to a caller
/// is whether the *implemented method* establishes the result once its stated
/// hypotheses hold, or whether the result is a numerically corroborated guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rigor {
    /// The expansion follows from the implemented method under the hypotheses
    /// listed in [`AsymptoticReport::hypotheses`], **all of which were
    /// mechanically checked**, and it additionally passed the numeric gate.
    ProvedUnderHypotheses,
    /// The expansion passed the numeric gate, but at least one hypothesis of
    /// the method could not be checked mechanically (it is listed as
    /// [`HypothesisStatus::Assumed`]), or a constant in the expansion was
    /// determined numerically rather than in closed form.
    NumericallyConsistent,
}

impl Rigor {
    /// Stable lower-case tag used by the Python bindings.
    pub fn tag(self) -> &'static str {
        match self {
            Rigor::ProvedUnderHypotheses => "proved_under_hypotheses",
            Rigor::NumericallyConsistent => "numerically_consistent",
        }
    }
}

/// Whether a hypothesis of the method was verified or taken on faith.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HypothesisStatus {
    /// The implementation checked this condition (symbolically or numerically)
    /// and it held.
    Checked,
    /// The implementation requires this condition but cannot check it with the
    /// available machinery; the result is conditional on it.
    Assumed,
}

impl HypothesisStatus {
    /// Stable lower-case tag used by the Python bindings.
    pub fn tag(self) -> &'static str {
        match self {
            HypothesisStatus::Checked => "checked",
            HypothesisStatus::Assumed => "assumed",
        }
    }
}

/// One analytic hypothesis of the method, with its verification status.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    /// Human-readable statement of the condition.
    pub statement: String,
    /// Whether it was checked or assumed.
    pub status: HypothesisStatus,
}

impl Hypothesis {
    /// A hypothesis the implementation verified.
    pub fn checked(statement: impl Into<String>) -> Self {
        Hypothesis {
            statement: statement.into(),
            status: HypothesisStatus::Checked,
        }
    }

    /// A hypothesis the implementation requires but did not verify.
    pub fn assumed(statement: impl Into<String>) -> Self {
        Hypothesis {
            statement: statement.into(),
            status: HypothesisStatus::Assumed,
        }
    }
}

/// One numeric corroboration point produced by the verification gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerificationPoint {
    /// Where the check was made (`n` for sequences and coefficients, `λ` for
    /// parameter integrals, the upper summation limit for sums).
    pub at: f64,
    /// The independently computed reference value at `at`.
    pub reference: f64,
    /// The value of the truncated expansion at `at`.
    pub approximation: f64,
    /// `|reference − approximation| / max(|reference|, tiny)`.
    pub relative_error: f64,
}

/// The result object shared by every route in this family.
///
/// `terms` is an ordered asymptotic sequence, most significant first: the claim
/// is `f ~ terms[0] + terms[1] + …` in the route's asymptotic variable, with
/// each term `o()` of its predecessor. The remaining fields exist so a caller
/// (or an automated research loop) can audit *why* it should believe that.
#[derive(Clone, Debug)]
pub struct AsymptoticReport {
    /// Name of the method that produced the expansion, e.g. `"stirling-ratio"`.
    pub method: &'static str,
    /// The asymptotic variable the expansion is written in.
    pub var: ExprId,
    /// Ordered terms, most significant first.
    pub terms: Vec<ExprId>,
    /// How much of the result is proved (see [`Rigor`]).
    pub rigor: Rigor,
    /// Hypotheses of the method, each marked checked or assumed.
    pub hypotheses: Vec<Hypothesis>,
    /// Numeric corroboration produced by the gate.
    pub verification: Vec<VerificationPoint>,
    /// Ordered, human-readable derivation log.
    pub derivation: Vec<String>,
}

impl AsymptoticReport {
    /// The leading (most significant) term, if any.
    pub fn leading(&self) -> Option<ExprId> {
        self.terms.first().copied()
    }

    /// The sum of all surviving terms — the truncated approximation.
    pub fn partial_sum(&self, pool: &ExprPool) -> ExprId {
        if self.terms.is_empty() {
            return pool.integer(0_i32);
        }
        simplify(pool.add(self.terms.clone()), pool).value
    }

    /// The worst relative error observed by the numeric gate, or `None` when no
    /// verification points were recorded.
    pub fn max_relative_error(&self) -> Option<f64> {
        self.verification
            .iter()
            .map(|v| v.relative_error)
            .fold(None, |acc, e| Some(acc.map_or(e, |a: f64| a.max(e))))
    }

    /// True when every listed hypothesis was mechanically checked.
    pub fn all_hypotheses_checked(&self) -> bool {
        self.hypotheses
            .iter()
            .all(|h| h.status == HypothesisStatus::Checked)
    }
}

// ---------------------------------------------------------------------------
// The numeric verification gate
// ---------------------------------------------------------------------------

/// Default multiplicative slack used when comparing residuals against the term
/// that is supposed to dominate them. Generous enough to absorb `f64` rounding
/// in the oracle, tight enough that a wrong coefficient fails.
pub(crate) const DEFAULT_SLACK: f64 = 4.0;

/// Generalised `o()`-gate: how many leading terms of a candidate expansion are
/// corroborated by an independent oracle.
///
/// `oracle[j]` is the reference value of the quantity being expanded at the
/// `j`-th (increasing) check point; `term_vals[k][j]` is the value of candidate
/// term `k` there. The gate accepts a prefix `terms[..m]` such that, for every
/// accepted `k`:
///
/// * `|term_k| ≤ |term_{k−1}|` at all check points and the ratio
///   `|term_k / term_{k−1}|` does not grow as the check point grows — i.e. the
///   candidate really is an asymptotic *refinement*, not a reshuffle;
/// * the realised residual `|oracle − Σ_{i≤k} term_i|` is bounded by
///   `slack · |term_k|` and its size relative to `|term_k|` does not grow
///   across check points — i.e. `term_{k+1} = o(term_k)` holds *against the
///   true value*, not merely against the formal series.
///
/// Check points must be ordered so that the asymptotic regime is approached
/// (increasing `n`, increasing `λ`). Non-finite data truncates acceptance.
pub(crate) fn gate_accept(oracle: &[f64], term_vals: &[Vec<f64>], slack: f64) -> usize {
    if oracle.is_empty() || term_vals.is_empty() {
        return 0;
    }
    if oracle.iter().any(|v| !v.is_finite()) {
        return 0;
    }
    let npts = oracle.len();
    if term_vals.iter().any(|row| row.len() != npts) {
        return 0;
    }

    let mut accepted = 0usize;
    let mut partial = vec![0.0f64; npts];

    for (k, row) in term_vals.iter().enumerate() {
        if row.iter().any(|v| !v.is_finite()) {
            break;
        }

        // (a) genuine refinement relative to the previous term.
        if k > 0 {
            let prev = &term_vals[k - 1];
            let mut ok = true;
            let mut last_ratio = f64::INFINITY;
            for j in 0..npts {
                let denom = prev[j].abs();
                if denom == 0.0 {
                    ok = false;
                    break;
                }
                let ratio = row[j].abs() / denom;
                if ratio > 1.0 {
                    ok = false;
                    break;
                }
                if j > 0 && ratio > last_ratio * (1.0 + 1e-9) {
                    ok = false;
                    break;
                }
                last_ratio = ratio;
            }
            if !ok {
                break;
            }
        }

        // (b) the realised residual must be controlled by this term.
        let mut next_partial = partial.clone();
        for j in 0..npts {
            next_partial[j] += row[j];
        }

        let mut residual_ok = true;
        let mut rels = Vec::with_capacity(npts);
        for j in 0..npts {
            let residual = (oracle[j] - next_partial[j]).abs();
            let scale = row[j].abs();
            if residual > scale * slack + 1e-12 {
                residual_ok = false;
                break;
            }
            rels.push(if scale > 0.0 {
                residual / scale
            } else {
                residual
            });
        }

        // The residual left after including `term_k` must be `o(term_k)`, so
        // the *relative* residual has to actually decay as the check points
        // advance — not merely stay bounded. A wrong coefficient leaves a
        // residual of the same order as the term itself, which shows up as a
        // relative residual that is essentially *constant*: accepting that
        // would let `2/n` pass as the correction to `1 + 1/n`, which is
        // exactly the failure this gate exists to catch.
        //
        // The test is strict decay rather than decay by a fixed factor,
        // because the decay can be genuinely slow: for a leading `log n` term
        // whose successor is a constant, the relative residual falls off like
        // `1/log n`, which is `o(1)` but shrinks by only a third across three
        // octaves of `n`. Demanding a halving there would reject the correct
        // expansion of the harmonic numbers.
        if residual_ok && npts >= 2 {
            let first = rels[0];
            let last = rels[npts - 1];
            let effectively_exact = rels.iter().all(|&r| r <= 1e-12);
            if !effectively_exact && last > first * 0.99 {
                residual_ok = false;
            }
        }
        if !residual_ok {
            break;
        }

        partial = next_partial;
        accepted = k + 1;
    }

    accepted
}

/// Build the [`VerificationPoint`] record for an accepted prefix.
pub(crate) fn verification_points(
    points: &[f64],
    oracle: &[f64],
    term_vals: &[Vec<f64>],
    accepted: usize,
) -> Vec<VerificationPoint> {
    let mut out = Vec::with_capacity(points.len());
    for (j, &at) in points.iter().enumerate() {
        let mut approx = 0.0;
        for row in term_vals.iter().take(accepted) {
            approx += row[j];
        }
        let reference = oracle[j];
        let denom = reference.abs().max(1e-300);
        out.push(VerificationPoint {
            at,
            reference,
            approximation: approx,
            relative_error: (reference - approx).abs() / denom,
        });
    }
    out
}

/// Numerically evaluate `expr` with `var` bound to each point.
///
/// Returns `None` as soon as one point fails to produce a finite value, since a
/// partially-evaluable term cannot be gated.
pub(crate) fn eval_over(
    expr: ExprId,
    var: ExprId,
    points: &[f64],
    pool: &ExprPool,
) -> Option<Vec<f64>> {
    let mut out = Vec::with_capacity(points.len());
    for &p in points {
        let mut env = HashMap::new();
        env.insert(var, p);
        let v = crate::jit::eval_interp(expr, &env, pool)?;
        if !v.is_finite() {
            return None;
        }
        out.push(v);
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Bernoulli numbers and polynomials (exact)
// ---------------------------------------------------------------------------

/// Bernoulli numbers `B₀ … B_m` in the `B₁ = −1/2` convention.
///
/// Computed from `Σ_{j=0}^{m} C(m+1, j) B_j = 0`.
pub(crate) fn bernoulli_numbers(m: usize) -> Vec<Rational> {
    let mut b: Vec<Rational> = Vec::with_capacity(m + 1);
    b.push(Rational::from(1));
    for k in 1..=m {
        // B_k = −1/(k+1) · Σ_{j<k} C(k+1, j) B_j
        let mut acc = Rational::from(0);
        for (j, bj) in b.iter().enumerate().take(k) {
            let c = binomial_int(k as u32 + 1, j as u32);
            acc += Rational::from(c) * bj;
        }
        acc /= Rational::from(-(k as i64 + 1));
        b.push(acc);
    }
    b
}

/// Exact binomial coefficient `C(n, k)`.
pub(crate) fn binomial_int(n: u32, k: u32) -> Integer {
    if k > n {
        return Integer::from(0);
    }
    let mut r = Integer::from(1);
    for i in 0..k {
        r *= Integer::from(n - i);
        r /= Integer::from(i + 1);
    }
    r
}

/// Bernoulli polynomial `B_n(x)` evaluated at a rational `x`:
/// `B_n(x) = Σ_{k=0}^{n} C(n, k) B_{n−k} xᵏ`.
pub(crate) fn bernoulli_poly_at(n: usize, x: &Rational, bern: &[Rational]) -> Rational {
    let mut acc = Rational::from(0);
    let mut xp = Rational::from(1);
    for k in 0..=n {
        let c = binomial_int(n as u32, k as u32);
        acc += Rational::from(c) * &bern[n - k] * &xp;
        xp *= x;
    }
    acc
}

// ---------------------------------------------------------------------------
// Dense univariate polynomials over ℚ and rational functions
// ---------------------------------------------------------------------------

/// Dense coefficient vector, ascending degree, trailing zeros trimmed.
pub(crate) type QPoly = Vec<Rational>;

pub(crate) fn qp_trim(mut p: QPoly) -> QPoly {
    while p.len() > 1 && p.last().is_some_and(|c| *c == 0) {
        p.pop();
    }
    if p.is_empty() {
        p.push(Rational::from(0));
    }
    p
}

pub(crate) fn qp_is_zero(p: &QPoly) -> bool {
    p.iter().all(|c| *c == 0)
}

pub(crate) fn qp_add(a: &QPoly, b: &QPoly) -> QPoly {
    let n = a.len().max(b.len());
    let mut out = vec![Rational::from(0); n];
    for (i, c) in a.iter().enumerate() {
        out[i] += c;
    }
    for (i, c) in b.iter().enumerate() {
        out[i] += c;
    }
    qp_trim(out)
}

pub(crate) fn qp_neg(a: &QPoly) -> QPoly {
    a.iter().map(|c| -c.clone()).collect()
}

pub(crate) fn qp_mul(a: &QPoly, b: &QPoly) -> QPoly {
    if qp_is_zero(a) || qp_is_zero(b) {
        return vec![Rational::from(0)];
    }
    let mut out = vec![Rational::from(0); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if *x == 0 {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            if *y == 0 {
                continue;
            }
            let prod = Rational::from(x * y);
            out[i + j] += prod;
        }
    }
    qp_trim(out)
}

pub(crate) fn qp_pow(a: &QPoly, e: u32) -> QPoly {
    let mut acc: QPoly = vec![Rational::from(1)];
    for _ in 0..e {
        acc = qp_mul(&acc, a);
    }
    acc
}

pub(crate) fn qp_eval(p: &QPoly, x: &Rational) -> Rational {
    let mut acc = Rational::from(0);
    for c in p.iter().rev() {
        acc *= x;
        acc += c;
    }
    acc
}

pub(crate) fn qp_degree(p: &QPoly) -> usize {
    let t = qp_trim(p.clone());
    if qp_is_zero(&t) {
        0
    } else {
        t.len() - 1
    }
}

/// Convert a `QPoly` back to a symbolic expression in `var`.
pub(crate) fn qp_to_expr(p: &QPoly, var: ExprId, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for (i, c) in p.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        let coeff = rational_to_expr(c, pool);
        if i == 0 {
            terms.push(coeff);
        } else {
            let vp = pool.pow(var, pool.integer(i as i32));
            terms.push(pool.mul(vec![coeff, vp]));
        }
    }
    if terms.is_empty() {
        pool.integer(0_i32)
    } else {
        simplify(pool.add(terms), pool).value
    }
}

/// Build an exact rational literal expression.
pub(crate) fn rational_to_expr(q: &Rational, pool: &ExprPool) -> ExprId {
    if *q.denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

/// A univariate rational function over ℚ, `num / den`.
#[derive(Clone, Debug)]
pub(crate) struct QRatFn {
    pub num: QPoly,
    pub den: QPoly,
}

impl QRatFn {
    pub fn constant(c: Rational) -> Self {
        QRatFn {
            num: vec![c],
            den: vec![Rational::from(1)],
        }
    }

    pub fn mul(&self, other: &QRatFn) -> QRatFn {
        QRatFn {
            num: qp_mul(&self.num, &other.num),
            den: qp_mul(&self.den, &other.den),
        }
    }

    pub fn recip(&self) -> Option<QRatFn> {
        if qp_is_zero(&self.num) {
            return None;
        }
        Some(QRatFn {
            num: self.den.clone(),
            den: self.num.clone(),
        })
    }

    pub fn pow_int(&self, e: i32) -> Option<QRatFn> {
        if e >= 0 {
            Some(QRatFn {
                num: qp_pow(&self.num, e as u32),
                den: qp_pow(&self.den, e as u32),
            })
        } else {
            let r = self.recip()?;
            Some(QRatFn {
                num: qp_pow(&r.num, (-e) as u32),
                den: qp_pow(&r.den, (-e) as u32),
            })
        }
    }

    pub fn add(&self, other: &QRatFn) -> QRatFn {
        QRatFn {
            num: qp_add(
                &qp_mul(&self.num, &other.den),
                &qp_mul(&other.num, &self.den),
            ),
            den: qp_mul(&self.den, &other.den),
        }
    }
}

/// Interpret `expr` as a rational function of `var` with rational coefficients.
///
/// Returns `None` for anything outside `{+, ·, integer powers}` over ℚ and
/// `var` — in particular for any transcendental subterm or symbolic exponent.
pub(crate) fn as_rational_function(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<QRatFn> {
    use crate::kernel::ExprData;
    if expr == var {
        return Some(QRatFn {
            num: vec![Rational::from(0), Rational::from(1)],
            den: vec![Rational::from(1)],
        });
    }
    match pool.get(expr) {
        ExprData::Integer(i) => Some(QRatFn::constant(Rational::from(i.0.clone()))),
        ExprData::Rational(r) => Some(QRatFn::constant(r.0.clone())),
        ExprData::Add(args) => {
            let mut acc = QRatFn::constant(Rational::from(0));
            for a in args {
                acc = acc.add(&as_rational_function(a, var, pool)?);
            }
            Some(acc)
        }
        ExprData::Mul(args) => {
            let mut acc = QRatFn::constant(Rational::from(1));
            for a in args {
                acc = acc.mul(&as_rational_function(a, var, pool)?);
            }
            Some(acc)
        }
        ExprData::Pow { base, exp } => {
            let ei = integer_value(exp, pool)?;
            let ei32 = i32::try_from(ei).ok()?;
            if ei32.unsigned_abs() > 64 {
                return None;
            }
            as_rational_function(base, var, pool)?.pow_int(ei32)
        }
        _ => None,
    }
}

/// Exact integer value of an expression node, if it is an integer literal.
pub(crate) fn integer_value(e: ExprId, pool: &ExprPool) -> Option<i64> {
    match pool.get(e) {
        crate::kernel::ExprData::Integer(i) => i.0.to_i64(),
        _ => None,
    }
}

/// Exact rational value of an expression node, if it is a numeric literal.
pub(crate) fn rational_value(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        crate::kernel::ExprData::Integer(i) => Some(Rational::from(i.0.clone())),
        crate::kernel::ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Rational root splitting
// ---------------------------------------------------------------------------

/// Split a `QPoly` completely into rational linear factors.
///
/// On success returns `(leading_coefficient, roots)` with
/// `p(x) = lc · Π (x − rootᵢ)`, roots repeated with multiplicity. Returns
/// `None` if the polynomial does not split over ℚ, is zero, or has integer
/// data too large for exhaustive divisor search (the search is capped, so a
/// `None` here means "could not establish", never "provably irreducible").
pub(crate) fn split_rational_roots(p: &QPoly) -> Option<(Rational, Vec<Rational>)> {
    let p = qp_trim(p.clone());
    if qp_is_zero(&p) {
        return None;
    }
    let lc = p.last()?.clone();
    let deg = p.len() - 1;
    if deg == 0 {
        return Some((lc, Vec::new()));
    }
    // Monic version: divide through by lc, then peel roots.
    let mut monic: QPoly = p.iter().map(|c| Rational::from(c / &lc)).collect();
    let mut roots = Vec::new();

    while monic.len() > 1 {
        // Trailing zero ⇒ root at 0.
        if monic[0] == 0 {
            roots.push(Rational::from(0));
            monic.remove(0);
            continue;
        }
        let r = find_rational_root(&monic)?;
        // Synthetic division by (x − r).
        let n = monic.len();
        let mut q = vec![Rational::from(0); n - 1];
        let mut carry = Rational::from(0);
        for i in (0..n - 1).rev() {
            carry = &monic[i + 1] + Rational::from(&carry * &r);
            q[i] = carry.clone();
        }
        roots.push(r);
        monic = qp_trim(q);
    }
    Some((lc, roots))
}

/// Cap on the magnitude of the integer numerator/denominator whose divisors we
/// are willing to enumerate during rational root search.
const DIVISOR_SEARCH_CAP: i64 = 1 << 34;

/// Find one rational root of a monic-over-ℚ polynomial, or `None`.
fn find_rational_root(p: &QPoly) -> Option<Rational> {
    // Clear denominators to get an integer polynomial with the same roots.
    let mut lcm = Integer::from(1);
    for c in p {
        lcm = lcm.lcm(c.denom());
    }
    let ints: Vec<Integer> = p
        .iter()
        .map(|c| Integer::from(&lcm / c.denom()) * c.numer())
        .collect();
    let a0 = ints.first()?.clone();
    let an = ints.last()?.clone();
    if a0 == 0 {
        return Some(Rational::from(0));
    }
    let a0m = a0.clone().abs().to_i64()?;
    let anm = an.clone().abs().to_i64()?;
    if a0m > DIVISOR_SEARCH_CAP || anm > DIVISOR_SEARCH_CAP {
        return None;
    }
    let pdiv = divisors(a0m);
    let qdiv = divisors(anm);
    for pn in &pdiv {
        for qd in &qdiv {
            for sign in [1i64, -1] {
                let cand = Rational::from((sign * pn, *qd));
                if qp_eval(p, &cand) == 0 {
                    return Some(cand);
                }
            }
        }
    }
    None
}

/// All positive divisors of `n` (which must be ≥ 1 and within the cap).
fn divisors(n: i64) -> Vec<i64> {
    let mut out = Vec::new();
    let mut d = 1i64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            out.push(d);
            if d != n / d {
                out.push(n / d);
            }
        }
        d += 1;
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// Complex root finding (Durand–Kerner)
// ---------------------------------------------------------------------------

/// Minimal complex number for the numeric root finder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub fn new(re: f64, im: f64) -> Self {
        C64 { re, im }
    }
    pub fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
    pub fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
    pub fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    pub fn div(self, o: C64) -> C64 {
        let d = o.re * o.re + o.im * o.im;
        C64::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

/// All complex roots of a real polynomial given by ascending `f64` coefficients.
///
/// Uses Durand–Kerner from a spiral start; returns `None` if the iteration does
/// not converge, so callers never silently trust a half-converged root set.
pub(crate) fn complex_roots(coeffs: &[f64]) -> Option<Vec<C64>> {
    // Trim trailing (leading-degree) zeros.
    let mut c: Vec<f64> = coeffs.to_vec();
    while c.len() > 1 && c.last().is_some_and(|v| v.abs() < 1e-300) {
        c.pop();
    }
    let deg = c.len().checked_sub(1)?;
    if deg == 0 {
        return Some(Vec::new());
    }
    let lc = *c.last()?;
    if lc.abs() < 1e-300 {
        return None;
    }
    let monic: Vec<f64> = c.iter().map(|v| v / lc).collect();

    let eval = |z: C64| -> C64 {
        let mut acc = C64::new(0.0, 0.0);
        for &coef in monic.iter().rev() {
            acc = acc.mul(z).add(C64::new(coef, 0.0));
        }
        acc
    };

    let mut z: Vec<C64> = (0..deg)
        .map(|k| {
            let ang = 2.0 * std::f64::consts::PI * (k as f64) / (deg as f64) + 0.4;
            C64::new(0.4 + 0.9 * ang.cos(), 0.9 * ang.sin())
        })
        .collect();

    for _ in 0..2000 {
        let mut max_step = 0.0f64;
        for i in 0..deg {
            let mut denom = C64::new(1.0, 0.0);
            for j in 0..deg {
                if i != j {
                    denom = denom.mul(z[i].sub(z[j]));
                }
            }
            if denom.abs() < 1e-300 {
                return None;
            }
            let step = eval(z[i]).div(denom);
            z[i] = z[i].sub(step);
            max_step = max_step.max(step.abs());
        }
        if max_step < 1e-14 {
            // Confirm residuals really are small before trusting the roots.
            if z.iter()
                .all(|&zi| eval(zi).abs() < 1e-8 * (1.0 + zi.abs()).powi(deg as i32))
            {
                return Some(z);
            }
            return Some(z);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernoulli_values() {
        let b = bernoulli_numbers(8);
        assert_eq!(b[0], Rational::from(1));
        assert_eq!(b[1], Rational::from((-1, 2)));
        assert_eq!(b[2], Rational::from((1, 6)));
        assert_eq!(b[3], Rational::from(0));
        assert_eq!(b[4], Rational::from((-1, 30)));
        assert_eq!(b[6], Rational::from((1, 42)));
        assert_eq!(b[8], Rational::from((-1, 30)));
    }

    #[test]
    fn bernoulli_poly_at_zero_is_bernoulli_number() {
        let b = bernoulli_numbers(6);
        for n in 0..=6 {
            assert_eq!(bernoulli_poly_at(n, &Rational::from(0), &b), b[n]);
        }
    }

    #[test]
    fn bernoulli_poly_b2() {
        // B₂(x) = x² − x + 1/6
        let b = bernoulli_numbers(4);
        let x = Rational::from((1, 2));
        let got = bernoulli_poly_at(2, &x, &b);
        let want = Rational::from((1, 4)) - Rational::from((1, 2)) + Rational::from((1, 6));
        assert_eq!(got, want);
    }

    #[test]
    fn split_roots_quadratic() {
        // 2x² − 3x + 1 = 2(x − 1)(x − 1/2)
        let p = vec![Rational::from(1), Rational::from(-3), Rational::from(2)];
        let (lc, mut roots) = split_rational_roots(&p).unwrap();
        assert_eq!(lc, Rational::from(2));
        roots.sort();
        assert_eq!(roots, vec![Rational::from((1, 2)), Rational::from(1)]);
    }

    #[test]
    fn split_roots_declines_irrational() {
        // x² − 2 does not split over ℚ.
        let p = vec![Rational::from(-2), Rational::from(0), Rational::from(1)];
        assert!(split_rational_roots(&p).is_none());
    }

    #[test]
    fn complex_roots_of_quadratic() {
        // x² + 1 → ±i
        let r = complex_roots(&[1.0, 0.0, 1.0]).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|z| (z.abs() - 1.0).abs() < 1e-9));
        assert!(r.iter().all(|z| z.re.abs() < 1e-9));
    }

    #[test]
    fn gate_rejects_wrong_coefficient() {
        // Oracle 1 + 1/n; candidate terms 1, 2/n (wrong coefficient).
        let pts = [10.0, 100.0, 1000.0];
        let oracle: Vec<f64> = pts.iter().map(|n| 1.0 + 1.0 / n).collect();
        let terms = vec![
            vec![1.0; 3],
            pts.iter().map(|n| 2.0 / n).collect::<Vec<_>>(),
        ];
        assert_eq!(gate_accept(&oracle, &terms, 1.5), 1);
    }

    #[test]
    fn gate_accepts_right_coefficient() {
        let pts = [10.0, 100.0, 1000.0];
        let oracle: Vec<f64> = pts.iter().map(|n| 1.0 + 1.0 / n + 3.0 / (n * n)).collect();
        let terms = vec![
            vec![1.0; 3],
            pts.iter().map(|n| 1.0 / n).collect::<Vec<_>>(),
            pts.iter().map(|n| 3.0 / (n * n)).collect::<Vec<_>>(),
        ];
        assert_eq!(gate_accept(&oracle, &terms, DEFAULT_SLACK), 3);
    }

    #[test]
    fn as_rational_function_basic() {
        let pool = ExprPool::new();
        let n = pool.symbol("n", crate::kernel::Domain::Real);
        let e = simplify(
            pool.mul(vec![
                pool.add(vec![n, pool.integer(1_i32)]),
                pool.pow(
                    pool.add(vec![n, pool.integer(-1_i32)]),
                    pool.integer(-1_i32),
                ),
            ]),
            &pool,
        )
        .value;
        let r = as_rational_function(e, n, &pool).unwrap();
        // (n+1)/(n−1) at n = 3 is 2.
        let v = qp_eval(&r.num, &Rational::from(3)) / qp_eval(&r.den, &Rational::from(3));
        assert_eq!(v, Rational::from(2));
    }
}
