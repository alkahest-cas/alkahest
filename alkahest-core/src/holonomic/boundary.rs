//! Deciding the boundary hypothesis a Zeilberger certificate rests on.
//!
//! [`super::zeilberger::zeilberger()`] proves an identity about the *summand*:
//!
//! ```text
//! Σ_i a_i(n)·F(n+i, k) = G(n, k+1) − G(n, k),    G(n,k) = R(n,k)·F(n,k)
//! ```
//!
//! Turning that into a statement about `S(n) = Σ_{k=κ₀(n)}^{κ₁(n)} F(n,k)` is a
//! second, separate step, and it is where a valid certificate can produce a
//! false theorem. This module takes it, three-valued, in the same discipline as
//! `verified_sign` / `verified_no_roots`:
//!
//! | [`BoundaryStatus`] | claim |
//! |---|---|
//! | [`Vanishes`](BoundaryStatus::Vanishes) | proved `b ≡ 0`; the homogeneous recurrence `Σ_i a_i(n)·S(n+i) = 0` holds for the sum |
//! | [`Nonzero`](BoundaryStatus::Nonzero) | `b(n)` computed exactly and proved `≢ 0`; the *inhomogeneous* `Σ_i a_i(n)·S(n+i) = b(n)` holds |
//! | [`Unknown`](BoundaryStatus::Unknown) | neither was established; **nothing** may be claimed about the sum |
//!
//! Only the first two are results. `Unknown` carries a reason, so a caller can
//! tell "the limits were not supplied" from "the certificate has a pole at the
//! endpoint" and act differently.
//!
//! # What `b(n)` actually is
//!
//! Summing the identity over `k = κ₀(n) .. κ₁(n)` telescopes the right-hand side
//! to `G(n, κ₁(n)+1) − G(n, κ₀(n))`. That is *not* the whole of `b(n)` whenever
//! the limits move with `n`, because the left-hand side then contains
//! `Σ_{k=κ₀(n)}^{κ₁(n)} F(n+i, k)`, which is **not** `S(n+i)` — `S(n+i)` runs
//! over `κ₀(n+i) .. κ₁(n+i)`. The missing pieces are finitely many explicit
//! values of `F`:
//!
//! ```text
//! b(n) = G(n, κ₁(n)+1) − G(n, κ₀(n)) + Σ_i a_i(n)·D_i(n)
//! D_i(n) = Σ_{k=κ₀(n+i)}^{κ₀(n)−1} F(n+i,k) + Σ_{k=κ₁(n)+1}^{κ₁(n+i)} F(n+i,k)
//! ```
//!
//! Dropping `D_i` is not a small error. For the textbook `Σ_{k=0}^{n} C(n,k)`
//! the telescoped part alone is `−1`, and it is `D_1 = C(n+1,n+1) = 1` that
//! cancels it: the correct verdict for the classical natural-boundary identities
//! comes out of the *sum*, not out of the endpoints on their own.
//!
//! # Why `Vanishes` is a proof
//!
//! Nothing here is numeric. Every piece above is evaluated as an exact
//! hypergeometric-in-`n` value `q(n)·βⁿ·∏ Γ(a_j·n + c_j)^{e_j}` with
//! `q ∈ Q(n)`:
//!
//! 1. `G(n, ·)` is meromorphic in `k`, and the telescoping identity is an
//!    identity of meromorphic functions, checked exactly in `Q(n)(k)` before the
//!    certificate is handed back. Given that `S` is a well-defined finite sum
//!    and `G(n, κ₀)` is finite, the identity at successive integers forces every
//!    intermediate `G` to be finite and telescopes exactly.
//! 2. An endpoint value is obtained by **order counting**, not substitution:
//!    with `k = k* + ε`, the order of the rational part is computed by exact
//!    deflation of the root `k*` over `Q(n)`, and every `Γ(a·n + b·k + c)^e`
//!    factor whose argument at `k*` is an `n`-free non-positive integer
//!    contributes `−e` (a simple pole of `Γ`, or a simple zero of `1/Γ`) with an
//!    exact residue `(−1)^m/(m!·b)`. A strictly positive total order means the
//!    value **is** `0`, exactly, for every `n`; a negative one means `G` is
//!    unbounded there, which breaks step 1 and is reported as `Unknown`.
//! 3. The resulting terms are put in a canonical form — `Γ(x+1) = x·Γ(x)` is
//!    applied until every argument is `a·n + c` with `c ∈ [0,1)`, the excess
//!    folded into `q` — and terms with identical `(β, Γ-signature)` are
//!    collected. `Vanishes` is returned only when **every** collected
//!    coefficient is the zero element of `Q(n)`. That is an identity of
//!    functions, not a sampled agreement.
//!
//! [`Nonzero`](BoundaryStatus::Nonzero) is symmetric in rigour: it needs a
//! *witness*, an integer `n₀` at which `b(n₀)` evaluates in exact rational
//! arithmetic to something other than zero. Sampling that finds only zeros
//! proves nothing, and yields `Unknown` — never `Vanishes`.
//!
//! # The residual hypothesis
//!
//! `q ∈ Q(n)` may have poles, and a `Γ` argument may land on a non-positive
//! integer, at isolated `n`. A verdict is therefore a statement about the
//! integers `n` at which the summand, the certificate and the sum are all
//! defined — stated rather than hidden, in
//! [`BoundaryStatus::side_conditions`].

use super::hyperterm::{affine_parts, as_ratk, rn_to_expr, GammaFactor, ProperTerm};
use super::qfield::{
    rn_add, rn_div, rn_eval, rn_int, rn_is_zero, rn_mul, rn_neg, rn_one, rn_rat, rn_var, rn_zero,
    PolyK, RatK, Rn,
};
use super::zeilberger::ZeilbergerResult;
use crate::kernel::{ExprId, ExprPool};
use rug::{Integer, Rational};

/// The verdict on `b(n)`, the inhomogeneity of the recurrence for the *sum*.
///
/// See the [module documentation](self) for what each variant is allowed to
/// mean and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundaryStatus {
    /// Proved `b ≡ 0`: the homogeneous recurrence `Σ_i a_i(n)·S(n+i) = 0` holds
    /// for the sum over the range that was supplied.
    Vanishes,
    /// `b(n)` was computed exactly and proved not identically zero. The
    /// **inhomogeneous** recurrence `Σ_i a_i(n)·S(n+i) = rhs(n)` holds — still a
    /// theorem about the sum, just not the homogeneous one.
    Nonzero {
        /// `b(n)`, as an expression in `n` alone.
        rhs: ExprId,
        /// An integer `n₀` at which `b(n₀) ≠ 0` was checked in exact rational
        /// arithmetic — the witness that `rhs` is not identically zero.
        witness_n: i64,
    },
    /// Neither could be established. **No** recurrence for the sum follows; the
    /// certificate remains a true statement about the summand and nothing more.
    Unknown {
        /// Why the verdict could not be reached, phrased so a caller can decide
        /// whether to supply better limits or to close the branch.
        reason: String,
    },
}

impl BoundaryStatus {
    /// `"vanishes"`, `"nonzero"` or `"unknown"` — the stable tag to record.
    pub fn tag(&self) -> &'static str {
        match self {
            BoundaryStatus::Vanishes => "vanishes",
            BoundaryStatus::Nonzero { .. } => "nonzero",
            BoundaryStatus::Unknown { .. } => "unknown",
        }
    }

    /// Whether a recurrence for the *sum* may be read off at all — true for
    /// [`Vanishes`](BoundaryStatus::Vanishes) (homogeneous) and
    /// [`Nonzero`](BoundaryStatus::Nonzero) (inhomogeneous), false for
    /// [`Unknown`](BoundaryStatus::Unknown).
    pub fn implies_sum_recurrence(&self) -> bool {
        !matches!(self, BoundaryStatus::Unknown { .. })
    }

    /// What is still assumed after this verdict, as plain strings — the same
    /// shape as `DerivedResult.verification["side_conditions"]`.
    ///
    /// This is *not* a fixed string: a discharged hypothesis and an open one
    /// read differently, which is the whole point of computing a verdict rather
    /// than restating the caveat. `range` is the summation range the verdict is
    /// about, e.g. `"k = 0..n"`.
    pub fn side_conditions(&self, range: &str) -> Vec<String> {
        match self {
            BoundaryStatus::Vanishes => vec![format!(
                "the boundary difference for {range} was proved to vanish in exact arithmetic, \
                 so the homogeneous recurrence sum_i a_i(n)*S(n+i) = 0 holds for the sum; it is \
                 a statement about the integers n at which the summand, the certificate and the \
                 sum are all defined"
            )],
            BoundaryStatus::Nonzero { witness_n, .. } => vec![format!(
                "the boundary difference for {range} does not vanish, so the recurrence for the \
                 sum is the inhomogeneous sum_i a_i(n)*S(n+i) = b(n) with b(n) returned \
                 explicitly (b({witness_n}) != 0 in exact arithmetic); the HOMOGENEOUS \
                 recurrence is FALSE for this sum"
            )],
            BoundaryStatus::Unknown { reason } => vec![format!(
                "the boundary difference for {range} could not be decided ({reason}); the \
                 certificate proves the telescoping identity in k and NOTHING follows about \
                 sum_k F(n,k) until this is discharged independently"
            )],
        }
    }
}

/// The summation range `k = 0 .. n` — the convention `Σ_{k=0}^{n}` that the
/// classical identities and the OEIS formula field both use.
///
/// It is a *default*, not an inference: [`boundary_status`] takes the limits
/// explicitly and returns [`BoundaryStatus::Unknown`] when they are not
/// supplied, so a caller who assumes this convention does so on the record.
pub fn natural_limits(n: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    (pool.integer(0_i32), n)
}

/// Decide the boundary hypothesis for `result` over `k = limits.0 .. limits.1`.
///
/// `term` must be the same `F(n,k)` that produced `result`. `limits` is
/// `(k_lo, k_hi)` and **`None` yields [`BoundaryStatus::Unknown`]** rather than
/// a guessed range, because guessing it silently is exactly the defect this
/// function exists to remove. Callers that mean `Σ_{k=0}^{n}` say so with
/// [`natural_limits`].
///
/// Both endpoints must be integer-affine in `n` (`α·n + β` with `α, β ∈ Z`).
/// Anything else — a second symbol, an infinity, a non-integer offset — is
/// reported as [`BoundaryStatus::Unknown`], which is the honest answer for a
/// range this analysis cannot place.
pub fn boundary_status(
    result: &ZeilbergerResult,
    term: ExprId,
    n: ExprId,
    k: ExprId,
    limits: Option<(ExprId, ExprId)>,
    pool: &ExprPool,
) -> BoundaryStatus {
    match collect_terms(result, term, n, k, limits, pool) {
        Err(reason) => BoundaryStatus::Unknown { reason },
        Ok(terms) => decide(&terms, n, pool),
    }
}

/// Assemble every signed piece of `b(n)`; see the module docs for the formula.
fn collect_terms(
    result: &ZeilbergerResult,
    term: ExprId,
    n: ExprId,
    k: ExprId,
    limits: Option<(ExprId, ExprId)>,
    pool: &ExprPool,
) -> Result<Vec<HypTerm>, String> {
    let Some((lo, hi)) = limits else {
        return Err(
            "the summation limits were not supplied, so there is no range over which \
                    to evaluate the boundary; pass (k_lo, k_hi) — the usual choice is \
                    k = 0..n"
                .into(),
        );
    };

    let f = ProperTerm::parse(term, n, k, pool)
        .map_err(|_| "the summand did not re-parse as a proper hypergeometric term".to_string())?;
    let r = as_ratk(result.certificate, n, k, pool, 0)
        .ok_or("the certificate did not re-parse as an element of Q(n)(k)")?;

    let lo_pt = endpoint_point(lo, n, k, pool).map_err(|e| format!("lower limit k_lo: {e}"))?;
    let hi_pt = endpoint_point(hi, n, k, pool).map_err(|e| format!("upper limit k_hi: {e}"))?;

    let order = result.order;
    let extras = (lo_pt.alpha.unsigned_abs() + hi_pt.alpha.unsigned_abs()) * order as u64;
    if extras > MAX_CORRECTION_TERMS {
        return Err(format!(
            "the summation limits move with n fast enough to need {extras} correction terms, \
             past the supported limit of {MAX_CORRECTION_TERMS}"
        ));
    }

    let mut terms: Vec<HypTerm> = Vec::new();

    // The telescoped part: + G(n, k_hi+1) − G(n, k_lo).
    let at_hi = hi_pt.offset(1);
    push_value(&mut terms, value_at(&r, &f, 0, at_hi), &rn_one())
        .map_err(|e| format!("G(n, k_hi+1) could not be evaluated: {e}"))?;
    push_value(&mut terms, value_at(&r, &f, 0, lo_pt), &rn_neg(&rn_one()))
        .map_err(|e| format!("G(n, k_lo) could not be evaluated: {e}"))?;

    // The range-shift corrections: Σ_i a_i(n)·D_i(n).
    let one = RatK::one();
    for i in 0..=order {
        let a_i = coeff_as_rn(result.coeffs[i], n, k, pool)
            .ok_or_else(|| format!("recurrence coefficient a_{i}(n) is not an element of Q(n)"))?;
        if rn_is_zero(&a_i) {
            continue;
        }
        let i64_i = i as i64;
        // Upper window: Σ_{k=κ₁(n)+1}^{κ₁(n+i)} F(n+i, k).
        for (t, sign) in signed_window(1, hi_pt.alpha * i64_i) {
            let weight = scale_sign(&a_i, sign);
            let at = hi_pt.offset(t);
            push_value(&mut terms, value_at(&one, &f, i64_i, at), &weight)
                .map_err(|e| format!("the upper range-shift correction failed: {e}"))?;
        }
        // Lower window: Σ_{k=κ₀(n+i)}^{κ₀(n)−1} F(n+i, k).
        for (t, sign) in signed_window(lo_pt.alpha * i64_i, -1) {
            let weight = scale_sign(&a_i, sign);
            let at = lo_pt.offset(t);
            push_value(&mut terms, value_at(&one, &f, i64_i, at), &weight)
                .map_err(|e| format!("the lower range-shift correction failed: {e}"))?;
        }
    }
    Ok(terms)
}

/// Group the pieces into hypergeometric-similarity classes and read off the
/// verdict.
fn decide(terms: &[HypTerm], n: ExprId, pool: &ExprPool) -> BoundaryStatus {
    let mut classes: Vec<(Rational, Vec<GammaN>, Rn)> = Vec::new();
    for t in terms {
        let Some((coeff, base, sig)) = t.canonical() else {
            return BoundaryStatus::Unknown {
                reason: "a boundary term could not be put in canonical hypergeometric form".into(),
            };
        };
        match classes.iter_mut().find(|(b, g, _)| *b == base && *g == sig) {
            Some((_, _, acc)) => *acc = rn_add(acc, &coeff),
            None => classes.push((base, sig, coeff)),
        }
    }
    // Every similarity class has a zero coefficient in Q(n): b ≡ 0, as an
    // identity of functions.
    if classes.iter().all(|(_, _, c)| rn_is_zero(c)) {
        return BoundaryStatus::Vanishes;
    }

    // Otherwise b(n) is explicit — but "the classes did not cancel" is not a
    // proof that b ≢ 0, because two classes can still be equal as functions.
    // Reporting `Nonzero` needs a witness.
    match nonzero_witness(terms) {
        Some(witness_n) => BoundaryStatus::Nonzero {
            // Report the *collected* classes: one summand per similarity class,
            // with the Γ ladder already worked off, which is both shorter and
            // free of the cancelling Γ(n+2)/Γ(n+1) pairs the raw terms carry.
            rhs: classes_expr(&classes, n, pool),
            witness_n,
        },
        None => BoundaryStatus::Unknown {
            reason: "the boundary difference is an explicit closed form that neither cancelled \
                     structurally nor evaluated to anything nonzero at a sampled integer n; it \
                     may well vanish, but that was not proved"
                .into(),
        },
    }
}

// ---------------------------------------------------------------------------
// Endpoints and windows
// ---------------------------------------------------------------------------

/// Correction terms are `O(slope · order)`; the cap only exists so that a
/// pathological range cannot ask for an unbounded expansion.
const MAX_CORRECTION_TERMS: u64 = 256;

/// A summation limit `κ(m) = α·m + β` with integer `α, β`.
#[derive(Debug, Clone, Copy)]
struct Point {
    alpha: i64,
    beta: i64,
}

impl Point {
    fn offset(self, t: i64) -> Point {
        Point {
            alpha: self.alpha,
            beta: self.beta + t,
        }
    }
}

/// Read a limit expression as `α·n + β`.
fn endpoint_point(e: ExprId, n: ExprId, k: ExprId, pool: &ExprPool) -> Result<Point, String> {
    let Some((a, b, c)) = affine_parts(e, n, k, pool) else {
        return Err(format!(
            "`{}` is not an integer-affine function of n",
            pool.display(e)
        ));
    };
    if b != 0 {
        return Err(format!(
            "`{}` depends on the summation index k",
            pool.display(e)
        ));
    }
    if *c.clone().denom() != 1 {
        return Err(format!(
            "`{}` has a non-integer constant part",
            pool.display(e)
        ));
    }
    let beta = c
        .numer()
        .to_i64()
        .ok_or_else(|| "the constant part is too large".to_string())?;
    Ok(Point { alpha: a, beta })
}

/// The integer offsets in `Σ_{t=from}^{to}`, with `Σ_{t=from}^{to} = −Σ_{t=to+1}^{from−1}`
/// when the range runs backwards — so a limit that *decreases* with `n` is
/// handled with the right sign rather than silently dropped.
fn signed_window(from: i64, to: i64) -> Vec<(i64, i32)> {
    if from <= to {
        (from..=to).map(|t| (t, 1)).collect()
    } else {
        ((to + 1)..=(from - 1)).map(|t| (t, -1)).collect()
    }
}

fn scale_sign(a: &Rn, sign: i32) -> Rn {
    if sign < 0 {
        rn_neg(a)
    } else {
        a.clone()
    }
}

/// A recurrence coefficient `a_i(n)` back as an element of `Q(n)`.
fn coeff_as_rn(e: ExprId, n: ExprId, k: ExprId, pool: &ExprPool) -> Option<Rn> {
    let r = as_ratk(e, n, k, pool, 0)?;
    if r.num.degree() > 0 || r.den.degree() > 0 {
        return None; // depends on k — not a recurrence coefficient
    }
    rn_div(&r.num.coeff(0), &r.den.coeff(0))
}

// ---------------------------------------------------------------------------
// Exact hypergeometric-in-n values
// ---------------------------------------------------------------------------

/// `Γ(a·n + c)^e`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GammaN {
    a: i64,
    c: Rational,
    e: i32,
}

/// `coeff(n) · base^n · ∏_j Γ(a_j·n + c_j)^{e_j}` — one exact summand of `b(n)`.
#[derive(Clone, Debug)]
struct HypTerm {
    coeff: Rn,
    base: Rational,
    gammas: Vec<GammaN>,
}

enum Value {
    /// Proved exactly zero.
    Zero,
    /// Finite, with this closed form.
    Finite(HypTerm),
    /// Not decidable by this analysis.
    Undecidable(String),
}

fn push_value(out: &mut Vec<HypTerm>, v: Value, weight: &Rn) -> Result<(), String> {
    match v {
        Value::Zero => Ok(()),
        Value::Undecidable(why) => Err(why),
        Value::Finite(mut t) => {
            t.coeff = rn_mul(&t.coeff, weight);
            if !rn_is_zero(&t.coeff) {
                out.push(t);
            }
            Ok(())
        }
    }
}

/// Largest `|e|` on a `Γ` factor, largest `Γ` argument, and largest integer part
/// shifted out during canonicalisation. All are far above anything a real
/// certificate produces; they bound the work a pathological input can ask for.
const MAX_GAMMA_EXPONENT: i32 = 64;
const MAX_GAMMA_ARG: i64 = 4096;
const MAX_GAMMA_SHIFT: i64 = 512;

/// Evaluate `extra(n,k) · F(n + n_shift, k)` at `k = α·n + β`, by order counting.
fn value_at(extra: &RatK, f: &ProperTerm, n_shift: i64, at: Point) -> Value {
    let q = extra.mul(&f.rat.shift_n(n_shift)).normalize();
    if q.num.is_zero() {
        return Value::Zero;
    }
    let kstar = rn_add(&rn_mul(&rn_int(at.alpha), &rn_var()), &rn_int(at.beta));

    // Order of the rational part at k*, by exact deflation over Q(n).
    let (m_num, num_r) = deflate_root(&q.num, &kstar);
    let (m_den, den_r) = deflate_root(&q.den, &kstar);
    let mut order: i64 = m_num as i64 - m_den as i64;

    let num_val = poly_eval(&num_r, &kstar);
    let den_val = poly_eval(&den_r, &kstar);
    if rn_is_zero(&den_val) {
        return Value::Undecidable(
            "the rational part's denominator vanishes identically at the endpoint".into(),
        );
    }
    let Some(mut coeff) = rn_div(&num_val, &den_val) else {
        return Value::Undecidable("the rational part is not invertible at the endpoint".into());
    };

    // z^k at k = α·n + β is (z^α)^n · z^β, and w^{n+shift} = w^n · w^shift.
    let (Some(z_alpha), Some(z_beta), Some(w_shift)) = (
        rat_pow(&f.z, at.alpha),
        rat_pow(&f.z, at.beta),
        rat_pow(&f.w, n_shift),
    ) else {
        return Value::Undecidable("an exponential factor has an unmanageable exponent".into());
    };
    let base = z_alpha * f.w.clone();
    coeff = rn_mul(&coeff, &rn_rat(z_beta * w_shift));

    let mut gammas: Vec<GammaN> = Vec::with_capacity(f.gammas.len());
    for g in &f.gammas {
        if g.e.unsigned_abs() > MAX_GAMMA_EXPONENT as u32 {
            return Value::Undecidable("a gamma factor has an unmanageable exponent".into());
        }
        match gamma_at_point(g, n_shift, at) {
            GammaAt::Finite { a, c } => gammas.push(GammaN { a, c, e: g.e }),
            GammaAt::Pole { residue_scale } => {
                // Γ(−m + b·ε) = (−1)^m / (m!·b·ε) · (1 + O(ε)): the factor
                // contributes ε-order −e and this exact leading coefficient.
                let Some(lead) = rat_pow(&residue_scale, g.e as i64) else {
                    return Value::Undecidable(
                        "a gamma residue has an unmanageable exponent".into(),
                    );
                };
                coeff = rn_mul(&coeff, &rn_rat(lead));
                order -= g.e as i64;
            }
            GammaAt::Degenerate(why) => return Value::Undecidable(why),
        }
    }

    if order > 0 {
        // A strictly positive order is an exact zero, for every n.
        return Value::Zero;
    }
    if order < 0 {
        return Value::Undecidable(format!(
            "the value is unbounded there (pole of order {}), so the telescoped sum has no \
             finite boundary value",
            -order
        ));
    }
    if rn_is_zero(&coeff) {
        return Value::Zero;
    }
    Value::Finite(HypTerm {
        coeff,
        base,
        gammas,
    })
}

enum GammaAt {
    /// `Γ(a·n + c)`, finite and nonzero for all but finitely many `n`.
    Finite { a: i64, c: Rational },
    /// The argument crosses a pole of `Γ` transversally in `k`; the leading
    /// coefficient of `Γ ~ residue_scale / ε` is exact.
    Pole { residue_scale: Rational },
    /// The factor is `∞` independently of `k`, so the term is not defined.
    Degenerate(String),
}

/// `Γ(a·(n+s) + b·k + c)` at `k = α·n + β`, i.e. `Γ((a + b·α)·n + (a·s + b·β + c))`.
fn gamma_at_point(g: &GammaFactor, n_shift: i64, at: Point) -> GammaAt {
    let Some(a_eff) = g.b.checked_mul(at.alpha).and_then(|v| v.checked_add(g.a)) else {
        return GammaAt::Degenerate("a gamma argument overflowed".into());
    };
    let Some(shift) =
        g.b.checked_mul(at.beta)
            .and_then(|v| g.a.checked_mul(n_shift).and_then(|w| v.checked_add(w)))
    else {
        return GammaAt::Degenerate("a gamma argument overflowed".into());
    };
    let c_eff = Rational::from(shift) + g.c.clone();

    let at_gamma_pole = a_eff == 0 && *c_eff.clone().denom() == 1 && c_eff <= 0;
    if !at_gamma_pole {
        return GammaAt::Finite { a: a_eff, c: c_eff };
    }
    if g.b == 0 {
        // No k-dependence at all: Γ of a fixed non-positive integer is a plain
        // infinity, so the term is not well defined here.
        return GammaAt::Degenerate(
            "the summand has a gamma factor at a pole independent of k".into(),
        );
    }
    let Some(m) = (-c_eff)
        .numer()
        .to_u64()
        .filter(|m| *m <= MAX_GAMMA_ARG as u64)
    else {
        return GammaAt::Degenerate("a gamma pole index is too large".into());
    };
    let sign = if m % 2 == 0 { 1 } else { -1 };
    let residue_scale = Rational::from(sign) / (Rational::from(factorial(m)) * Rational::from(g.b));
    GammaAt::Pole { residue_scale }
}

impl HypTerm {
    /// Canonical hypergeometric form: `Γ(x+1) = x·Γ(x)` applied until every
    /// argument is `a·n + c` with `c ∈ [0,1)`, the excess folded into the
    /// coefficient, equal arguments merged and the result sorted.
    ///
    /// Two terms are *similar* — their ratio is a rational function, so they can
    /// legitimately be added — exactly when they agree on `(base, signature)`.
    fn canonical(&self) -> Option<(Rn, Rational, Vec<GammaN>)> {
        let mut coeff = self.coeff.clone();
        let mut sig: Vec<GammaN> = Vec::with_capacity(self.gammas.len());
        for g in &self.gammas {
            let m = floor_rat(&g.c);
            let m_i64 = m.to_i64().filter(|v| v.abs() <= MAX_GAMMA_SHIFT)?;
            let c0 = g.c.clone() - Rational::from(m);
            if g.a == 0 && c0 == 0 {
                // Γ of a positive integer m: fold the factorial in outright.
                let value = Rational::from(factorial(u64::try_from(m_i64 - 1).ok()?));
                coeff = rn_mul(&coeff, &rn_rat(rat_pow(&value, g.e as i64)?));
                continue;
            }
            // Γ(a·n + c0 + m) = Γ(a·n + c0) · ∏_{t=0}^{m-1} (a·n + c0 + t)  (m ≥ 0)
            //                 = Γ(a·n + c0) / ∏_{t=m}^{-1} (a·n + c0 + t)   (m < 0)
            let mut ladder = rn_one();
            let range = if m_i64 >= 0 { 0..m_i64 } else { m_i64..0 };
            for t in range {
                let factor = rn_add(
                    &rn_mul(&rn_int(g.a), &rn_var()),
                    &rn_rat(c0.clone() + Rational::from(t)),
                );
                ladder = rn_mul(&ladder, &factor);
            }
            let ladder = if m_i64 >= 0 {
                ladder
            } else {
                super::qfield::rn_inv(&ladder)?
            };
            coeff = rn_mul(&coeff, &rn_pow(&ladder, g.e)?);
            sig.push(GammaN {
                a: g.a,
                c: c0,
                e: g.e,
            });
        }
        sig.sort_by(|x, y| x.a.cmp(&y.a).then_with(|| x.c.cmp(&y.c)));
        let mut merged: Vec<GammaN> = Vec::with_capacity(sig.len());
        for g in sig {
            match merged.last_mut() {
                Some(prev) if prev.a == g.a && prev.c == g.c => {
                    prev.e = prev.e.checked_add(g.e)?;
                }
                _ => merged.push(g),
            }
        }
        merged.retain(|g| g.e != 0);
        Some((coeff, self.base.clone(), merged))
    }

    /// Exact value at `n = n₀`, or `None` when it cannot be pinned down there
    /// (a pole of the coefficient, a non-integer `Γ` argument, a `Γ` infinity in
    /// the numerator).
    fn eval(&self, n0: i64) -> Option<Rational> {
        let mut acc = rn_eval(&self.coeff, &Rational::from(n0))?;
        acc *= rat_pow(&self.base, n0)?;
        let mut vanishes = false;
        for g in &self.gammas {
            let arg = Rational::from(g.a) * Rational::from(n0) + g.c.clone();
            if *arg.clone().denom() != 1 {
                return None;
            }
            let m = arg.numer().to_i64().filter(|m| *m <= MAX_GAMMA_ARG)?;
            if m <= 0 {
                // Γ(m) is infinite: 1/Γ contributes an exact zero, Γ itself an
                // infinity this evaluation cannot resolve.
                if g.e > 0 {
                    return None;
                }
                vanishes = true;
                continue;
            }
            acc *= rat_pow(&Rational::from(factorial((m - 1) as u64)), g.e as i64)?;
        }
        Some(if vanishes { Rational::from(0) } else { acc })
    }
}

// ---------------------------------------------------------------------------
// The witness for `Nonzero`
// ---------------------------------------------------------------------------

/// Integers `n` at which `b(n)` is evaluated in exact rational arithmetic while
/// looking for a witness that it is not identically zero.
///
/// Small on purpose: a witness is normally found at the first usable point, and
/// the values involve factorials of `O(n)`. Finding none is reported as
/// `Unknown`, never as "it vanishes".
const WITNESS_POINTS: std::ops::RangeInclusive<i64> = 1..=16;

fn nonzero_witness(terms: &[HypTerm]) -> Option<i64> {
    'points: for n0 in WITNESS_POINTS {
        let mut acc = Rational::from(0);
        for t in terms {
            let Some(v) = t.eval(n0) else {
                continue 'points;
            };
            acc += v;
        }
        if acc != 0 {
            return Some(n0);
        }
    }
    None
}

/// `b(n)` as one summand per similarity class.
fn classes_expr(classes: &[(Rational, Vec<GammaN>, Rn)], n: ExprId, pool: &ExprPool) -> ExprId {
    let mut parts: Vec<ExprId> = Vec::with_capacity(classes.len());
    for (base, gammas, coeff) in classes {
        if rn_is_zero(coeff) {
            continue;
        }
        let mut factors = vec![rn_to_expr(pool, n, coeff)];
        if *base != 1 {
            factors.push(pool.pow(rational_expr(pool, base), n));
        }
        for g in gammas {
            let gam = pool.func("gamma", vec![gamma_arg_expr(pool, n, g)]);
            factors.push(if g.e == 1 {
                gam
            } else {
                pool.pow(gam, pool.integer(g.e))
            });
        }
        parts.push(pool.mul(factors));
    }
    let sum = match parts.len() {
        0 => pool.integer(0_i32),
        1 => parts[0],
        _ => pool.add(parts),
    };
    crate::simplify::simplify(sum, pool).value
}

// ---------------------------------------------------------------------------
// Expression helpers
// ---------------------------------------------------------------------------

fn gamma_arg_expr(pool: &ExprPool, n: ExprId, g: &GammaN) -> ExprId {
    let mut parts = Vec::with_capacity(2);
    if g.a != 0 {
        parts.push(if g.a == 1 {
            n
        } else {
            pool.mul(vec![pool.integer(g.a), n])
        });
    }
    if g.c != 0 || parts.is_empty() {
        parts.push(rational_expr(pool, &g.c));
    }
    if parts.len() == 1 {
        parts[0]
    } else {
        pool.add(parts)
    }
}

fn rational_expr(pool: &ExprPool, q: &Rational) -> ExprId {
    if *q.clone().denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

// ---------------------------------------------------------------------------
// Small exact helpers
// ---------------------------------------------------------------------------

/// Divide `p` by `k − root` once: quotient and remainder, both over `Q(n)`.
fn synthetic_div(p: &PolyK, root: &Rn) -> (PolyK, Rn) {
    let d = p.degree();
    if d <= 0 {
        return (PolyK::zero(), p.coeff(0));
    }
    let d = d as usize;
    let mut q = vec![rn_zero(); d];
    q[d - 1] = p.coeff(d);
    for i in (1..d).rev() {
        q[i - 1] = rn_add(&p.coeff(i), &rn_mul(root, &q[i]));
    }
    let rem = rn_add(&p.coeff(0), &rn_mul(root, &q[0]));
    (PolyK::from_coeffs(q), rem)
}

/// Multiplicity of `root` in `p`, and `p` with that many factors removed.
fn deflate_root(p: &PolyK, root: &Rn) -> (usize, PolyK) {
    let mut cur = p.clone();
    let mut m = 0usize;
    while !cur.is_zero() {
        let (quot, rem) = synthetic_div(&cur, root);
        if !rn_is_zero(&rem) {
            break;
        }
        cur = quot;
        m += 1;
    }
    (m, cur)
}

/// Horner evaluation of a `Q(n)[k]` polynomial at a point of `Q(n)`.
fn poly_eval(p: &PolyK, x: &Rn) -> Rn {
    let d = p.degree();
    if d < 0 {
        return rn_zero();
    }
    let mut acc = p.coeff(d as usize);
    for i in (0..d as usize).rev() {
        acc = rn_add(&rn_mul(&acc, x), &p.coeff(i));
    }
    acc
}

fn rn_pow(a: &Rn, e: i32) -> Option<Rn> {
    if e == 0 {
        return Some(rn_one());
    }
    let base = if e < 0 {
        super::qfield::rn_inv(a)?
    } else {
        a.clone()
    };
    let mut acc = rn_one();
    for _ in 0..e.unsigned_abs() {
        acc = rn_mul(&acc, &base);
    }
    Some(acc)
}

/// Largest `|exponent|` accepted when raising an exact rational to a power.
const MAX_RAT_POW: u64 = 4096;

fn rat_pow(q: &Rational, e: i64) -> Option<Rational> {
    if e == 0 {
        return Some(Rational::from(1));
    }
    if e.unsigned_abs() > MAX_RAT_POW {
        return None;
    }
    if *q == 0 {
        return if e > 0 { Some(Rational::from(0)) } else { None };
    }
    let base = if e < 0 { q.clone().recip() } else { q.clone() };
    let mut acc = Rational::from(1);
    for _ in 0..e.unsigned_abs() {
        acc *= base.clone();
    }
    Some(acc)
}

fn factorial(m: u64) -> Integer {
    let mut acc = Integer::from(1);
    for t in 2..=m {
        acc *= t;
    }
    acc
}

/// `⌊q⌋` — `rug`'s integer division truncates toward zero, which is not the same
/// thing for negative arguments, and the `Γ(x+1) = x·Γ(x)` ladder needs the
/// floor.
fn floor_rat(q: &Rational) -> Integer {
    let num = q.numer().clone();
    let den = q.denom().clone();
    let mut quot = num.clone() / den.clone();
    if num < 0 && quot.clone() * den != num {
        quot -= 1;
    }
    quot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holonomic::zeilberger::{zeilberger, ZeilbergerOpts};
    use crate::kernel::Domain;

    fn nk(pool: &ExprPool) -> (ExprId, ExprId) {
        (
            pool.symbol("n", Domain::Real),
            pool.symbol("k", Domain::Real),
        )
    }

    /// `C(top, bot)` as the ratio of gammas the parser recognises.
    fn binom(pool: &ExprPool, top: ExprId, bot: ExprId) -> ExprId {
        let one = pool.integer(1_i32);
        let g1 = pool.func("gamma", vec![pool.add(vec![top, one])]);
        let g2 = pool.func("gamma", vec![pool.add(vec![bot, one])]);
        let g3 = pool.func(
            "gamma",
            vec![pool.add(vec![top, pool.mul(vec![bot, pool.integer(-1_i32)]), one])],
        );
        pool.mul(vec![
            g1,
            pool.pow(g2, pool.integer(-1_i32)),
            pool.pow(g3, pool.integer(-1_i32)),
        ])
    }

    fn verdict(
        f: ExprId,
        n: ExprId,
        k: ExprId,
        pool: &ExprPool,
        limits: Option<(ExprId, ExprId)>,
    ) -> BoundaryStatus {
        let cert = zeilberger(f, n, k, pool, &ZeilbergerOpts::default()).expect("certificate");
        boundary_status(&cert.value, f, n, k, limits, pool)
    }

    /// The classical natural-boundary case: `Σ_{k=0}^{n} C(n,k) = 2ⁿ`.
    ///
    /// The telescoped part alone is `−1` here; it is the range-shift correction
    /// `a_1·C(n+1,n+1)` that cancels it. Getting `Vanishes` is therefore a test
    /// of the *whole* formula, not just of the endpoints.
    #[test]
    fn natural_boundary_is_proved_to_vanish() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let f = binom(&pool, n, k);
        let status = verdict(f, n, k, &pool, Some(natural_limits(n, &pool)));
        assert_eq!(status, BoundaryStatus::Vanishes, "got {status:?}");
        assert!(status.implies_sum_recurrence());
        assert_eq!(status.tag(), "vanishes");
    }

    /// Franel `Σ_{k=0}^{n} C(n,k)³` — order 2, still a natural boundary.
    #[test]
    fn franel_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let f = pool.mul(vec![c, c, c]);
        let status = verdict(f, n, k, &pool, Some(natural_limits(n, &pool)));
        assert_eq!(status, BoundaryStatus::Vanishes, "got {status:?}");
    }

    /// Dixon `Σ_{k=0}^{n} (−1)^k C(n,k)³` — the `z^k` factor must not disturb
    /// the endpoint analysis.
    #[test]
    fn dixon_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let sign = pool.pow(pool.integer(-1_i32), k);
        let f = pool.mul(vec![sign, c, c, c]);
        let status = verdict(f, n, k, &pool, Some(natural_limits(n, &pool)));
        assert_eq!(status, BoundaryStatus::Vanishes, "got {status:?}");
    }

    /// Apéry `Σ_{k=0}^{n} C(n,k)²·C(n+k,k)²` — the heaviest natural-boundary
    /// case, and the one whose gamma arguments genuinely move with `n`.
    #[test]
    fn apery_boundary_vanishes() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let d = binom(&pool, pool.add(vec![n, k]), k);
        let f = pool.mul(vec![c, c, d, d]);
        let status = verdict(f, n, k, &pool, Some(natural_limits(n, &pool)));
        assert_eq!(status, BoundaryStatus::Vanishes, "got {status:?}");
    }

    /// `F = C(n,k)/(k+1)`, the textbook counterexample: the homogeneous
    /// recurrence is false and the verdict must say so *with* the
    /// inhomogeneity, not merely refuse.
    ///
    /// `S(n) = (2ⁿ⁺¹−1)/(n+1)`, which pins `b(n)` end to end.
    #[test]
    fn nonvanishing_boundary_returns_a_usable_inhomogeneous_recurrence() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let kp1 = pool.add(vec![k, pool.integer(1_i32)]);
        let f = pool.mul(vec![
            binom(&pool, n, k),
            pool.pow(kp1, pool.integer(-1_i32)),
        ]);
        let cert = zeilberger(f, n, k, &pool, &ZeilbergerOpts::default()).expect("certificate");
        let status = boundary_status(&cert.value, f, n, k, Some(natural_limits(n, &pool)), &pool);
        let BoundaryStatus::Nonzero { rhs, .. } = &status else {
            panic!("expected a nonzero boundary, got {status:?}");
        };
        assert_eq!(status.tag(), "nonzero");
        assert!(status.side_conditions("k = 0..n")[0].contains("FALSE"));

        // Σ_i a_i(n)·S(n+i) = b(n), checked against the closed form.
        let s = |m: f64| (2.0_f64.powf(m + 1.0) - 1.0) / (m + 1.0);
        for ni in 2..7 {
            let env = std::collections::HashMap::from([(n, ni as f64)]);
            let lhs: f64 = cert
                .value
                .coeffs
                .iter()
                .enumerate()
                .map(|(i, &c)| {
                    crate::eval_f64(c, &pool, &env).expect("a_i(n)") * s((ni + i) as f64)
                })
                .sum();
            let b = crate::eval_f64(*rhs, &pool, &env).expect("b(n) evaluates");
            assert!(
                (lhs - b).abs() < 1e-6 * lhs.abs().max(1.0),
                "at n = {ni}: Σ a_i S(n+i) = {lhs} but b(n) = {b}"
            );
            assert!(b.abs() > 1e-9, "b({ni}) must not be zero");
        }
    }

    /// Missing limits are the unsafe default, so they are refused outright.
    #[test]
    fn absent_limits_are_unknown_not_assumed() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let f = binom(&pool, n, k);
        let status = verdict(f, n, k, &pool, None);
        let BoundaryStatus::Unknown { reason } = &status else {
            panic!("expected unknown, got {status:?}");
        };
        assert!(reason.contains("limits"), "{reason}");
        assert!(!status.implies_sum_recurrence());
        assert!(status.side_conditions("an unspecified range")[0].contains("NOTHING"));
    }

    /// An endpoint this analysis cannot place is `unknown`, never `vanishes` —
    /// on the very summand that *is* proved to vanish over `k = 0..n`.
    #[test]
    fn a_range_that_cannot_be_placed_is_unknown() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let m = pool.symbol("m", Domain::Real);
        let f = binom(&pool, n, k);
        let status = verdict(f, n, k, &pool, Some((pool.integer(0_i32), m)));
        assert!(
            matches!(status, BoundaryStatus::Unknown { .. }),
            "got {status:?}"
        );
    }

    /// Truncating the range one term early turns the same certificate from a
    /// homogeneous recurrence into an inhomogeneous one. This is the limits
    /// design earning its keep: the verdict is about the range, not only about
    /// the summand.
    #[test]
    fn truncating_the_range_changes_the_verdict() {
        let pool = ExprPool::new();
        let (n, k) = nk(&pool);
        let c = binom(&pool, n, k);
        let f = pool.mul(vec![c, c, c]);
        let cert = zeilberger(f, n, k, &pool, &ZeilbergerOpts::default()).expect("certificate");
        let full = boundary_status(&cert.value, f, n, k, Some(natural_limits(n, &pool)), &pool);
        assert_eq!(full, BoundaryStatus::Vanishes, "got {full:?}");

        let n_minus_1 = pool.add(vec![n, pool.integer(-1_i32)]);
        let short = boundary_status(
            &cert.value,
            f,
            n,
            k,
            Some((pool.integer(0_i32), n_minus_1)),
            &pool,
        );
        assert!(
            matches!(short, BoundaryStatus::Nonzero { .. }),
            "dropping the k = n term leaves C(n,n)³ behind: got {short:?}"
        );
    }

    /// Deflation and Horner evaluation are the only two places the order count
    /// can go wrong, so they are pinned directly.
    #[test]
    fn root_deflation_counts_multiplicity_exactly() {
        // (k − n)³ · (k + 1) over Q(n)[k]
        let root = rn_var();
        let linear = PolyK::from_coeffs(vec![rn_neg(&root), rn_int(1)]);
        let other = PolyK::from_coeffs(vec![rn_int(1), rn_int(1)]);
        let p = linear.mul(&linear).mul(&linear).mul(&other);
        let (m, rest) = deflate_root(&p, &root);
        assert_eq!(m, 3);
        assert!(rest.eq_poly(&other));
        assert!(!rn_is_zero(&poly_eval(&rest, &root)));
    }

    /// A backwards window is a signed sum, not an empty one — the sign is what
    /// makes a limit that *decreases* with `n` come out right.
    #[test]
    fn signed_windows_carry_their_orientation() {
        assert_eq!(signed_window(1, 2), vec![(1, 1), (2, 1)]);
        assert_eq!(signed_window(1, 0), Vec::<(i64, i32)>::new());
        assert_eq!(signed_window(1, -1), vec![(0, -1)]);
        assert_eq!(signed_window(0, -1), Vec::<(i64, i32)>::new());
        assert_eq!(signed_window(-1, -1), vec![(-1, 1)]);
    }

    /// `⌊·⌋`, not truncation — the `Γ` ladder is wrong by one for negative
    /// arguments otherwise.
    #[test]
    fn floor_is_a_floor() {
        assert_eq!(floor_rat(&Rational::from((7, 2))), Integer::from(3));
        assert_eq!(floor_rat(&Rational::from((-7, 2))), Integer::from(-4));
        assert_eq!(floor_rat(&Rational::from(-3)), Integer::from(-3));
        assert_eq!(floor_rat(&Rational::from(0)), Integer::from(0));
    }
}
