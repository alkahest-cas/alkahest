//! Definite sums whose bounds and summand share a parameter:
//! `Σ_{k=0}^{n} C(n,k) = 2ⁿ`, `Σ k·C(n,k) = n·2ⁿ⁻¹`, `Σ C(n,k)² = C(2n,n)`.
//!
//! Gosper cannot see these — the summand is not hypergeometric in `k` alone
//! once `n` is free, and no indefinite sum exists anyway — and they are the
//! textbook Zeilberger case. The route:
//!
//! 1. [`crate::holonomic::zeilberger()`] finds an exactly verified
//!    first-order creative-telescoping relation
//!    `a₁(n)·F(n+1,k) + a₀(n)·F(n,k) = G(n,k+1) − G(n,k)`;
//! 2. [`crate::holonomic::boundary_verdict`] decides — exactly, three-valued —
//!    what that relation says about the *sum* over the declared range:
//!    `a₁(n)·S(n+1) + a₀(n)·S(n) = b(n)` with `b ≡ 0` proved, or `b(n)` explicit;
//!    an undecided boundary is a refusal;
//! 3. the first-order recurrence is solved from an exact initial value
//!    `S(n₀)` by a definite product (`∏ −a₀/a₁`, Γ-ratios from
//!    [`super::product_definite`]), plus a Gosper sum for the inhomogeneous
//!    part. `rsolve` is *not* used: it solves constant-coefficient recurrences,
//!    and these have polynomial coefficients (`(n+1)·S(n+1) = 2(2n+1)·S(n)`);
//! 4. the closed form is compared with the sum, evaluated term by term in
//!    exact arithmetic, at every `n` from the start of its stated domain to
//!    `n₀ + VERIFY_PAST_N0`. Below `n₀` that comparison *is* the proof (the
//!    recurrence is only claimed from `n₀`, where `a₁` has no integer root
//!    left); above it, it guards the isolated `n` at which a certificate
//!    denominator could vanish. A disagreement is a refusal, never a value.
//!
//! The domain is stated, not hidden: the result carries `n ∈ ℤ` and `n ≥ n_start`
//! as side conditions, `n_start` being the smallest `n` from which every checked
//! value agrees (`0` for the classical identities).

use super::{simp, SumError};
use crate::deriv::log::SideCondition;
use crate::holonomic::{boundary_verdict, zeilberger, BoundaryStatus, ZeilbergerOpts};
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::orthopoly::rational_literal;
use rug::{Integer, Rational};
use std::collections::{HashMap, HashSet};

/// Largest number of terms summed directly to evaluate `S(n)` at one `n`.
const MAX_DIRECT_TERMS: i64 = 4096;
/// How far past `n₀` the closed form is checked against the direct sum.
const VERIFY_PAST_N0: i64 = 30;
/// Certificate degree bound for the first-order search.
const MAX_CERT_DEGREE: usize = 8;

pub(super) enum Outcome {
    Sum {
        value: ExprId,
        conditions: Vec<SideCondition>,
        rule: &'static str,
    },
    Refused(String),
    NotApplicable,
}

/// Free symbols of `e` other than the named constants.
fn free_symbols(e: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(x) = stack.pop() {
        if !seen.insert(x) {
            continue;
        }
        pool.with(x, |d| match d {
            ExprData::Symbol { .. } if !crate::eval::symbols::is_named_constant(x, pool) => {
                if !out.contains(&x) {
                    out.push(x)
                }
            }
            ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
                stack.extend(a.iter().copied())
            }
            ExprData::Pow { base, exp } => {
                stack.push(*base);
                stack.push(*exp);
            }
            _ => {}
        });
    }
}

fn as_int(pool: &ExprPool, e: ExprId) -> Option<i64> {
    let q = rational_literal(pool, e)?;
    if *q.denom() != 1 {
        return None;
    }
    q.numer().to_i64()
}

/// `S(m) = Σ_{k=lo(m)}^{hi(m)} F(m, k)` in exact arithmetic, or `None` when a
/// bound or a term does not evaluate to a rational number.
fn direct_sum(
    term: ExprId,
    n: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    m: i64,
    pool: &ExprPool,
) -> Option<Rational> {
    let at_n = |e: ExprId| {
        let mut map = HashMap::new();
        map.insert(n, pool.integer(m));
        simp(pool, subs(e, &map, pool))
    };
    let a = as_int(pool, at_n(lo))?;
    let b = as_int(pool, at_n(hi))?;
    if b < a {
        // An empty range (b = a − 1) is 0; a backwards one has no agreed value.
        return (b == a - 1).then(|| Rational::from(0));
    }
    if b - a >= MAX_DIRECT_TERMS {
        return None;
    }
    let mut total = Rational::from(0);
    for j in a..=b {
        let mut map = HashMap::new();
        map.insert(n, pool.integer(m));
        map.insert(k, pool.integer(j));
        let v = simp(pool, subs(term, &map, pool));
        // `C(1, 2) = Γ(2)/(Γ(3)·Γ(0))`: a reciprocal Γ at a pole is the natural
        // boundary of a binomial, and the term is 0.
        if vanishes_by_reciprocal_gamma(v, pool) {
            continue;
        }
        total += rational_literal(pool, v)?;
    }
    Some(total)
}

/// `Γ(m)` for an integer literal `m ≤ 0`.
fn is_gamma_pole(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Func { name, args } if name == "gamma" && args.len() == 1 => {
            rational_literal(pool, args[0]).is_some_and(|q| *q.denom() == 1 && q <= 0)
        }
        _ => false,
    }
}

/// Whether a denominator product contains a `Γ` pole.
fn has_gamma_pole_factor(e: ExprId, pool: &ExprPool) -> bool {
    if is_gamma_pole(e, pool) {
        return true;
    }
    match pool.get(e) {
        ExprData::Mul(xs) => xs.iter().any(|&x| has_gamma_pole_factor(x, pool)),
        ExprData::Pow { base, exp } => {
            rational_literal(pool, exp).is_some_and(|q| *q.denom() == 1 && q > 0)
                && has_gamma_pole_factor(base, pool)
        }
        _ => false,
    }
}

/// `e` is a product with a `Γ` pole in its denominator and none in its
/// numerator, so it is exactly `0`.
fn vanishes_by_reciprocal_gamma(e: ExprId, pool: &ExprPool) -> bool {
    let factors = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    };
    let in_denominator = |f: ExprId| match pool.get(f) {
        ExprData::Pow { base, exp } => {
            rational_literal(pool, exp).is_some_and(|q| *q.denom() == 1 && q < 0)
                && has_gamma_pole_factor(base, pool)
        }
        _ => false,
    };
    factors.iter().any(|&f| in_denominator(f))
        && !factors.iter().any(|&f| has_gamma_pole_factor(f, pool))
}

/// The closed form at `n = m`, as an exact rational.
fn closed_at(value: ExprId, n: ExprId, m: i64, pool: &ExprPool) -> Option<Rational> {
    let mut map = HashMap::new();
    map.insert(n, pool.integer(m));
    let v = simp(pool, subs(value, &map, pool));
    if let Some(q) = rational_literal(pool, v) {
        return Some(q);
    }
    let v = crate::specfun::gammasimp(v, pool).ok()?;
    rational_literal(pool, simp(pool, v))
}

/// Ascending rational coefficients of a polynomial expression in `n`.
fn poly_coeffs(e: ExprId, n: ExprId, pool: &ExprPool) -> Option<Vec<Rational>> {
    let rf = super::product::expr_to_ratfunc_pub(e, n, pool)?;
    if rf.den.degree() != 0 {
        return None;
    }
    let d = rf.den.coeffs[0].clone();
    Some(rf.num.coeffs.iter().map(|c| c.clone() / d.clone()).collect())
}

fn poly_eval(c: &[Rational], x: i64) -> Rational {
    let mut acc = Rational::from(0);
    for ci in c.iter().rev() {
        acc *= x;
        acc += ci;
    }
    acc
}

/// One past the largest integer root of the polynomial `c` (Cauchy's bound
/// limits the scan), or `i64::MIN` when it has none.
fn past_last_integer_root(c: &[Rational]) -> Option<i64> {
    let lead = c.last()?.clone();
    if lead == 0 {
        return None;
    }
    let mut bound = Rational::from(0);
    for ci in &c[..c.len() - 1] {
        let r = Rational::from((ci.clone() / lead.clone()).abs());
        if r > bound {
            bound = r;
        }
    }
    let b = (bound + 1u32).ceil().numer().to_i64()?;
    if b > 1_000_000 {
        return None;
    }
    let mut last = i64::MIN;
    for x in -b..=b {
        if poly_eval(c, x) == 0 {
            last = x + 1;
        }
    }
    Some(last)
}

/// `∏_{i=n₀}^{N−1} r(i)` for `r = −a₀/a₁` split into linear factors, as
/// `K^{N−n₀} · ∏ [Γ(α·N + β) / Γ(α·n₀ + β)]^e`.
///
/// Factors `a·i + c` with `a > 1` are grouped into complete residue blocks
/// `{c₀, …, c₀+a−1}` wherever the block's other members are present with the
/// same exponent or reduce (`gcd(a, c′) > 1`) to a smaller `a`: the block's
/// product telescopes to `Γ(a·N + c₀)/Γ(a·n₀ + c₀)` with an *integer* shift, so
/// `(2i+1)/(i+1)` gives `Γ(2N+1)/Γ(N+1)²` (the central binomial) rather than
/// `Γ(N+½)/Γ(N+1)`, and the closed form evaluates to a rational at every
/// integer — which is what the final check against the direct sum needs.
struct ProductPlan {
    /// Per-step constant `K`.
    k: Rational,
    /// `(α, β, e)`.
    gammas: Vec<(i64, Rational, i64)>,
}

impl ProductPlan {
    fn new(a0: &[Rational], a1: &[Rational], var: ExprId) -> Option<Self> {
        let poly = |c: &[Rational]| crate::matrix::normal_form::RatUniPoly { coeffs: c.to_vec() };
        let (k0, f0) = super::product::linear_factorization(&poly(a0), var)?;
        let (k1, f1) = super::product::linear_factorization(&poly(a1), var)?;
        let mut k = -(k0 / k1);
        // (a, c) → exponent, primitive keys only.
        let mut m: HashMap<(i64, i64), i64> = HashMap::new();
        for (a, c, e) in f0 {
            *m.entry((a.to_i64()?, c.to_i64()?)).or_insert(0) += e;
        }
        for (a, c, e) in f1 {
            *m.entry((a.to_i64()?, c.to_i64()?)).or_insert(0) -= e;
        }
        m.retain(|_, e| *e != 0);
        let gcd = |a: i64, b: i64| Integer::from(a).gcd(&Integer::from(b)).to_i64().unwrap_or(1);
        let mut gammas = Vec::new();
        let mut stuck: HashSet<(i64, i64)> = HashSet::new();
        loop {
            let pick = m
                .iter()
                .filter(|(&(a, _), _)| a > 1)
                .filter(|(key, _)| !stuck.contains(key))
                .map(|(&key, &e)| (key, e))
                .max();
            let Some(((a, c), e)) = pick else { break };
            // A block {c0, …, c0+a−1} containing c, preferring c0 = c.
            let block = (0..a).map(|s| c - s).find(|&c0| {
                (c0..c0 + a).all(|cp| cp == c || gcd(a, cp) > 1 || m.get(&(a, cp)) == Some(&e))
            });
            let Some(c0) = block else {
                stuck.insert((a, c));
                continue;
            };
            for cp in c0..c0 + a {
                let g = gcd(a, cp);
                if g > 1 {
                    // a·i + cp = g·((a/g)·i + cp/g), compensated.
                    *m.entry((a / g, cp / g)).or_insert(0) -= e;
                    for _ in 0..e.abs() {
                        if e > 0 {
                            k /= Rational::from(g);
                        } else {
                            k *= Rational::from(g);
                        }
                    }
                } else {
                    m.remove(&(a, cp));
                }
            }
            m.retain(|_, e| *e != 0);
            gammas.push((a, Rational::from(c0), e));
        }
        // Whatever is left: a·i + c = a·(i + c/a).
        for ((a, c), e) in m {
            for _ in 0..e.abs() {
                if e > 0 {
                    k *= Rational::from(a);
                } else {
                    k /= Rational::from(a);
                }
            }
            gammas.push((1, Rational::from((c, a)), e));
        }
        Some(ProductPlan { k, gammas })
    }

    /// The least `n₀` at which every `Γ(α·n₀ + β)` is finite and nonzero.
    fn min_start(&self) -> i64 {
        let mut lo = i64::MIN;
        for (alpha, beta, _) in &self.gammas {
            // α·n₀ + β > 0  ⇔  n₀ > −β/α.
            let bound = (Rational::from(-beta.clone()) / Rational::from(*alpha))
                .floor()
                .numer()
                .to_i64()
                .unwrap_or(i64::MAX - 1)
                + 1;
            lo = lo.max(bound);
        }
        lo
    }

    fn closed(&self, n0: i64, upper: ExprId, pool: &ExprPool) -> ExprId {
        // K^{N−n₀} as K^N · K^{−n₀}, so S(n₀)·K^{−n₀} folds into one rational
        // (`2·2^{n−1}` would otherwise be left unsimplified).
        let mut parts = Vec::new();
        if self.k != 1 {
            let k = crate::orthopoly::rat_node(pool, &self.k);
            parts.push(pool.pow(k, upper));
            parts.push(pool.pow(k, pool.integer(-n0)));
        }
        for (alpha, beta, e) in &self.gammas {
            let b = crate::orthopoly::rat_node(pool, beta);
            let arg = simp(pool, pool.add(vec![pool.mul(vec![pool.integer(*alpha), upper]), b]));
            let arg0 = crate::orthopoly::rat_node(
                pool,
                &(Rational::from(*alpha) * Rational::from(n0) + beta.clone()),
            );
            parts.push(pool.pow(pool.func("gamma", vec![arg]), pool.integer(*e)));
            parts.push(pool.pow(pool.func("gamma", vec![arg0]), pool.integer(-*e)));
        }
        simp(pool, pool.mul(parts))
    }
}

fn recurrence_text(coeffs: &[ExprId], pool: &ExprPool) -> String {
    format!(
        "({})·S(n+1) + ({})·S(n)",
        pool.display(coeffs[1]),
        pool.display(coeffs[0])
    )
}

/// See the module documentation.
pub(super) fn parametric_definite(
    term: ExprId,
    k: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<Outcome, SumError> {
    let mut bound_syms = Vec::new();
    free_symbols(lo, pool, &mut bound_syms);
    free_symbols(hi, pool, &mut bound_syms);
    bound_syms.retain(|&s| s != k);
    if bound_syms.is_empty() {
        return Ok(Outcome::NotApplicable);
    }
    let mut all = bound_syms.clone();
    free_symbols(term, pool, &mut all);
    all.retain(|&s| s != k);
    if all.len() != 1 {
        let names: Vec<String> = all.iter().map(|&s| pool.display(s).to_string()).collect();
        return Ok(Outcome::Refused(format!(
            "the bounds depend on a parameter and the summation involves {} free symbols ({}); \
             the Zeilberger route for parametric definite sums handles exactly one parameter",
            all.len(),
            names.join(", ")
        )));
    }
    let n = all[0];
    if !mentions_var(term, n, pool) {
        // Σ_{k=0}^{n} f(k): Gosper's territory, already tried.
        return Ok(Outcome::NotApplicable);
    }

    let opts = ZeilbergerOpts {
        max_order: 1,
        max_degree: MAX_CERT_DEGREE,
    };
    let z = match zeilberger(term, n, k, pool, &opts) {
        Ok(z) => z.value,
        Err(e) => {
            return Ok(Outcome::Refused(format!(
                "this is a definite sum whose summand depends on the parameter n = {} of its \
                 bounds (the Zeilberger case); no first-order creative-telescoping relation was \
                 found (certificate degree ≤ {MAX_CERT_DEGREE}): {e}. A sum whose recurrence has \
                 order 2 or more (Franel, Apéry) has no hypergeometric closed form found here",
                pool.display(n)
            )))
        }
    };
    let verdict = boundary_verdict(&z, term, n, k, Some((lo, hi)), pool);
    let rec = recurrence_text(&z.coeffs, pool);
    let rhs = match &verdict.status {
        BoundaryStatus::Vanishes => None,
        BoundaryStatus::Nonzero { rhs, .. } => Some(*rhs),
        BoundaryStatus::Unknown { reason } => {
            return Ok(Outcome::Refused(format!(
                "Zeilberger found {rec} = boundary terms for S(n) = Σ_k F(n,k), but the boundary \
                 terms over the range k = {}..{} could not be decided ({reason}), so no \
                 recurrence for the sum follows",
                pool.display(lo),
                pool.display(hi)
            )))
        }
    };
    if !verdict.certificate_poles.is_empty() {
        return Ok(Outcome::Refused(format!(
            "Zeilberger found {rec} = 0, but its certificate has poles inside the range"
        )));
    }

    // n₀: the verdict's domain, past every integer root of a₁ (where the
    // recurrence cannot be solved forward) and of a₀ (past which S is 0), and
    // where every Γ of the product is finite.
    let a1 = poly_coeffs(z.coeffs[1], n, pool);
    let a0 = poly_coeffs(z.coeffs[0], n, pool);
    let (Some(a1c), Some(a0c)) = (a1, a0) else {
        return Ok(Outcome::Refused(format!(
            "Zeilberger found {rec}, whose coefficients are not polynomials in n with rational \
             coefficients"
        )));
    };
    let domain_lo = verdict.valid_from.unwrap_or(0).max(0);
    // a₀ ≡ 0: the recurrence gives S(n+1) outright (`Σ (−1)^k C(n,k)`, whose
    // closed form 0 starts at n = 1).
    if a0c.iter().all(|c| *c == 0) {
        let Some(root1) = past_last_integer_root(&a1c) else {
            return Ok(Outcome::Refused(format!(
                "Zeilberger found {rec}; the integer roots of its leading coefficient could not \
                 be bounded"
            )));
        };
        let value = match rhs {
            None => pool.integer(0),
            Some(b) => {
                let nm1 = simp(pool, pool.add(vec![n, pool.integer(-1)]));
                let mut map = HashMap::new();
                map.insert(n, nm1);
                let quotient = pool.mul(vec![
                    subs(b, &map, pool),
                    pool.pow(subs(z.coeffs[1], &map, pool), pool.integer(-1)),
                ]);
                simp(pool, quotient)
            }
        };
        let n0 = domain_lo.max(1).max(root1) + 1;
        return Ok(verified(
            value,
            n0,
            domain_lo,
            (term, n, k, lo, hi),
            &rec,
            pool,
        ));
    }
    let (Some(root1), Some(root0)) = (past_last_integer_root(&a1c), past_last_integer_root(&a0c))
    else {
        return Ok(Outcome::Refused(format!(
            "Zeilberger found {rec}; the integer roots of its coefficients could not be bounded"
        )));
    };
    let Some(plan) = ProductPlan::new(&a0c, &a1c, n) else {
        return Ok(Outcome::Refused(format!(
            "Zeilberger found {rec}, but its coefficients do not split into linear factors \
             over Q, so the recurrence has no Γ-product solution here"
        )));
    };
    let n0 = domain_lo
        .max(1)
        .max(root1)
        .max(root0)
        .max(plan.min_start());

    let Some(s0) = direct_sum(term, n, k, lo, hi, n0, pool) else {
        return Ok(Outcome::Refused(format!(
            "Zeilberger found {rec}, but the initial value S({n0}) did not evaluate exactly"
        )));
    };

    // S(n) = P(n)·(S(n₀) + Σ_{j=n₀}^{n−1} c(j)/P(j+1)),  P(N) = ∏_{i=n₀}^{N−1} r(i),
    // r = −a₀/a₁, c = b/a₁.
    let at = |e: ExprId, v: ExprId| {
        let mut map = HashMap::new();
        map.insert(n, v);
        subs(e, &map, pool)
    };
    let inv = |e: ExprId| pool.pow(e, pool.integer(-1));
    let n0e = pool.integer(n0);
    let nm1 = simp(pool, pool.add(vec![n, pool.integer(-1)]));
    let p_n = plan.closed(n0, n, pool);
    let s0e = crate::orthopoly::rat_node(pool, &s0);
    let value = match rhs {
        None => simp(pool, pool.mul(vec![s0e, p_n])),
        Some(b) => {
            let j = pool.symbol("__zeil_j", Domain::Integer);
            let p_j1 = plan.closed(n0, pool.add(vec![j, pool.integer(1)]), pool);
            let c_j = pool.mul(vec![at(b, j), inv(at(z.coeffs[1], j))]);
            let summand = simp(pool, pool.mul(vec![c_j, inv(p_j1)]));
            let inner = match super::sum_definite(summand, j, n0e, nm1, pool) {
                Ok(d) => d.value,
                Err(e) => {
                    return Ok(Outcome::Refused(format!(
                        "Zeilberger found the inhomogeneous {rec} = {}; the particular \
                         solution needs Σ_j {} , which was refused: {e}",
                        pool.display(b),
                        pool.display(summand)
                    )))
                }
            };
            simp(pool, pool.mul(vec![p_n, pool.add(vec![s0e, inner])]))
        }
    };
    let value = match crate::specfun::gammasimp(value, pool) {
        Ok(v) => simp(pool, v),
        Err(_) => value,
    };

    Ok(verified(
        value,
        n0,
        domain_lo,
        (term, n, k, lo, hi),
        &rec,
        pool,
    ))
}

/// The final gate: every `n` from `n₀` to `n₀ + VERIFY_PAST_N0` must agree with
/// the direct sum, and the stated domain extends below `n₀` exactly as far as
/// agreement does.
fn verified(
    value: ExprId,
    n0: i64,
    domain_lo: i64,
    (term, n, k, lo, hi): (ExprId, ExprId, ExprId, ExprId, ExprId),
    rec: &str,
    pool: &ExprPool,
) -> Outcome {
    for m in n0..=n0 + VERIFY_PAST_N0 {
        let want = direct_sum(term, n, k, lo, hi, m, pool);
        let got = closed_at(value, n, m, pool);
        if want.is_none() || want != got {
            return Outcome::Refused(format!(
                "Zeilberger found {rec}; the closed form {} it gives disagrees with the sum at \
                 n = {m} (or one side did not evaluate exactly), so it is not returned",
                pool.display(value)
            ));
        }
    }
    let mut n_start = n0;
    while n_start > domain_lo {
        let m = n_start - 1;
        let want = direct_sum(term, n, k, lo, hi, m, pool);
        if want.is_none() || want != closed_at(value, n, m, pool) {
            break;
        }
        n_start = m;
    }
    let mut conditions = vec![SideCondition::InDomain(n, Domain::Integer)];
    // n ≥ n_start, stated as n − n_start + 1 > 0.
    conditions.push(SideCondition::Positive(simp(
        pool,
        pool.add(vec![n, pool.integer(1 - n_start)]),
    )));
    Outcome::Sum {
        value,
        conditions,
        rule: "zeilberger_first_order_recurrence",
    }
}
