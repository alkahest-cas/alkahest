//! Definite integrals that are Beta or incomplete-gamma functions (F11).
//!
//! These are the definite integrals whose antiderivative is not elementary
//! (or not elementary *with a symbolic exponent*), so the fundamental-theorem
//! route cannot reach them, while the definite value is a classical special
//! function:
//!
//! | integrand, interval | value | conditions |
//! |---|---|---|
//! | `(x−a)^p (b−x)^q`, `[a, b]` | `(b−a)^{p+q+1} B(p+1, q+1)` | `p > −1`, `q > −1` |
//! | `x^p (c+dx)^{−r}`, `[0, ∞)` | `c^{p+1−r} d^{−p−1} B(p+1, r−p−1)` | `c, d > 0`, `p > −1`, `r > p+1` |
//! | `sin^p x cos^q x`, `[0, π/2]` | `½ B((p+1)/2, (q+1)/2)` | `p > −1`, `q > −1` |
//! | `x^p e^{−r x^k}`, `[a, b] ⊂ [0, ∞]`, `k ∈ {1, 2}` | `r^{−s}/k · (Γ(s, r aᵏ) − Γ(s, r bᵏ))`, `s = (p+1)/k` | `r > 0`; `s > 0` if `a = 0` |
//!
//! Linear factors are matched up to a constant: `(2 − 2x)^q` on `[0, 1]` is
//! `2^q (1 − x)^q`. `B` is written as its `Γ` ratio (exact where the `Γ`
//! values are), the incomplete gamma functions as `uppergamma` /
//! `lowergamma`, whose own simplification gives the `erf`/`erfc` and
//! elementary forms at half-integer and integer orders.
//!
//! # Conditions
//!
//! A sign the route needs for the *formula* to apply (a rate `r > 0`, the
//! interval's orientation, a linear factor's slope) must be established — from
//! a numeric value, the symbol's domain or the active assumptions — or the
//! route declines and the ordinary route runs. A **convergence** condition on
//! an exponent (`p > −1`) is stated as a side condition when it is not
//! established, and a convergence condition that is established *false* is a
//! coded refusal (the integral diverges). Integrands with only non-negative
//! integer exponents are left to the fundamental-theorem route, which already
//! integrates them exactly.

use super::engine::{is_free_of, IntegrationError};
use super::prenorm::{poly_coeffs, sign_of};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep, SideCondition};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::Sign;

type Outcome = Option<Result<DerivedExpr<ExprId>, IntegrationError>>;

/// A product split as `(var-free coefficient factors, [(base, exponent)])`.
pub(super) type Factors = (Vec<ExprId>, Vec<(ExprId, ExprId)>);

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn mk_mul(mut f: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match f.len() {
        0 => pool.integer(1_i32),
        1 => f.pop().unwrap(),
        _ => pool.mul(f),
    }
}

fn mk_add(mut t: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    match t.len() {
        0 => pool.integer(0_i32),
        1 => t.pop().unwrap(),
        _ => pool.add(t),
    }
}

fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1_i32), e])
}

fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

/// `b^e`, folding `1^e = 1` (which the simplifier keeps for a symbolic `e`).
fn powr(b: ExprId, e: ExprId, pool: &ExprPool) -> ExprId {
    if b == pool.integer(1_i32) {
        b
    } else {
        pool.pow(b, e)
    }
}

fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

pub(super) fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    simp(e, pool) == pool.integer(0_i32)
}

pub(super) fn is_pos_inf(e: ExprId, pool: &ExprPool) -> bool {
    e == pool.pos_infinity()
}

fn has_infinity(e: ExprId, pool: &ExprPool) -> bool {
    if e == pool.pos_infinity() {
        return true;
    }
    match pool.get(e) {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().any(|&x| has_infinity(x, pool))
        }
        ExprData::Pow { base, exp } => has_infinity(base, pool) || has_infinity(exp, pool),
        _ => false,
    }
}

/// A finite bound: free of `var` and of `∞`.
fn finite_bound(b: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    is_free_of(b, var, pool) && !has_infinity(b, pool)
}

/// Is `e` a non-negative integer literal?
fn is_nonneg_int(e: ExprId, pool: &ExprPool) -> bool {
    match pool.get(e) {
        ExprData::Integer(n) => n.0 >= 0,
        _ => false,
    }
}

fn is_integer_literal(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(_))
}

pub(super) fn pi(pool: &ExprPool) -> ExprId {
    crate::eval::symbols::pi_symbol(pool)
}

/// `(α, β)` with `e = α·var + β`, `α ≠ 0`.
pub(super) fn linear(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if e == var {
        return Some((pool.integer(1_i32), pool.integer(0_i32)));
    }
    let c = poly_coeffs(e, var, pool)?;
    if c.len() != 2 || c[1] == pool.integer(0_i32) {
        return None;
    }
    Some((c[1], c[0]))
}

/// The factors of a product, `sqrt(u)` read as `u^{1/2}` and `u` as `u^1`:
/// `(var-free coefficient, [(base, exponent)])`. `None` when an exponent
/// depends on `var`.
pub(super) fn power_factors(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<Factors> {
    let factors = match pool.get(e) {
        ExprData::Mul(a) => a,
        _ => vec![e],
    };
    let mut coeff = Vec::new();
    let mut powers = Vec::new();
    for f in factors {
        if is_free_of(f, var, pool) {
            coeff.push(f);
            continue;
        }
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                if !is_free_of(exp, var, pool) {
                    return None;
                }
                // `sqrt(u)^e = u^{e/2}`.
                if let ExprData::Func { name, args } = pool.get(base) {
                    if name == "sqrt" && args.len() == 1 {
                        let half = pool.rational(1, 2);
                        powers.push((args[0], simp(pool.mul(vec![half, exp]), pool)));
                        continue;
                    }
                }
                // `(u^a)^n = u^{an}` for an integer `n`.
                if let ExprData::Pow {
                    base: inner,
                    exp: e1,
                } = pool.get(base)
                {
                    if matches!(pool.get(exp), ExprData::Integer(_)) && is_free_of(e1, var, pool) {
                        powers.push((inner, simp(pool.mul(vec![e1, exp]), pool)));
                        continue;
                    }
                }
                powers.push((base, exp));
            }
            ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => {
                powers.push((args[0], pool.rational(1, 2)));
            }
            _ => powers.push((f, pool.integer(1_i32))),
        }
    }
    Some((coeff, powers))
}

/// `Γ(v)` exactly where it has an exact value, else the node.
pub(super) fn gamma_of(v: ExprId, pool: &ExprPool) -> ExprId {
    crate::specfun::gamma::gamma_value(v, pool).unwrap_or_else(|| pool.func("gamma", vec![v]))
}

/// `B(a, b)` as an exact value where one exists, else the `Γ` ratio
/// `Γ(a)Γ(b)/Γ(a+b)`.
fn beta_closed(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    let a = simp(a, pool);
    let b = simp(b, pool);
    if let Some(v) = crate::specfun::incgamma::beta_value(a, b, pool) {
        return v;
    }
    let ab = simp(pool.add(vec![a, b]), pool);
    mk_mul(
        vec![
            gamma_of(a, pool),
            gamma_of(b, pool),
            pool.pow(gamma_of(ab, pool), pool.integer(-1_i32)),
        ],
        pool,
    )
}

/// The convergence condition `v > 0`: `Ok(None)` when established,
/// `Ok(Some(cond))` when it must be stated, `Err` when it is established false.
fn positive_condition(v: ExprId, pool: &ExprPool) -> Result<Option<SideCondition>, ExprId> {
    let v = simp(v, pool);
    match sign_of(v, pool) {
        Some(Sign::Positive) => Ok(None),
        Some(_) => Err(v),
        None => Ok(Some(SideCondition::Positive(v))),
    }
}

fn divergent(v: ExprId, pool: &ExprPool) -> IntegrationError {
    IntegrationError::NotImplemented(format!(
        "divergent integral: the convergence condition {} > 0 does not hold",
        pool.display(v)
    ))
}

/// Collect the convergence conditions; `Err` is the divergence refusal.
fn conditions(vs: &[ExprId], pool: &ExprPool) -> Result<Vec<SideCondition>, IntegrationError> {
    let mut out = Vec::new();
    for &v in vs {
        match positive_condition(v, pool) {
            Ok(Some(c)) => out.push(c),
            Ok(None) => {}
            Err(bad) => return Err(divergent(bad, pool)),
        }
    }
    Ok(out)
}

fn finish(
    rule: &'static str,
    expr: ExprId,
    value: ExprId,
    conds: Vec<SideCondition>,
    pool: &ExprPool,
) -> Result<DerivedExpr<ExprId>, IntegrationError> {
    let v = simplify(value, pool);
    let mut log = DerivationLog::new();
    log.push(RewriteStep::with_conditions(rule, expr, v.value, conds));
    Ok(DerivedExpr::with_log(v.value, log.merge(v.log)))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Try every route of this module and of [`super::feynman`] on
/// `∫_lower^upper expr d var` (bounds already simplified). `None` when no
/// route applies.
pub(crate) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Outcome {
    if !crate::kernel::expr_props::mult_tree_is_commutative(pool, expr) {
        return None;
    }
    if crate::budget::check().is_err() {
        return None;
    }
    if !is_free_of(lower, var, pool) || !is_free_of(upper, var, pool) {
        return None;
    }
    beta_interval(expr, var, lower, upper, pool)
        .or_else(|| beta_half_line(expr, var, lower, upper, pool))
        .or_else(|| beta_trig(expr, var, lower, upper, pool))
        .or_else(|| incomplete_gamma(expr, var, lower, upper, pool))
        .or_else(|| super::feynman::try_definite(expr, var, lower, upper, pool))
}

// ---------------------------------------------------------------------------
// Beta on a finite interval
// ---------------------------------------------------------------------------

/// `∫_a^b c·Π(αᵢx+βᵢ)^{eᵢ} dx` where every linear factor vanishes at `a` or
/// at `b`: `c·Π|αᵢ|^{eᵢ}·(b−a)^{p+q+1}·B(p+1, q+1)`.
fn beta_interval(expr: ExprId, var: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> Outcome {
    if !finite_bound(lo, var, pool) || !finite_bound(hi, var, pool) {
        return None;
    }
    let h = simp(sub(hi, lo, pool), pool);
    if sign_of(h, pool) != Some(Sign::Positive) {
        return None;
    }
    let (mut coeff, powers) = power_factors(expr, var, pool)?;
    let (mut p_parts, mut q_parts) = (Vec::new(), Vec::new());
    for (base, e) in powers {
        let (alpha, beta) = linear(base, var, pool)?;
        let root = simp(
            neg(
                pool.mul(vec![beta, pool.pow(alpha, pool.integer(-1_i32))]),
                pool,
            ),
            pool,
        );
        if is_zero(sub(root, lo, pool), pool) {
            // αx + β = α(x − a): needs α > 0 so that the power splits.
            if sign_of(alpha, pool) != Some(Sign::Positive) {
                return None;
            }
            coeff.push(powr(alpha, e, pool));
            p_parts.push(e);
        } else if is_zero(sub(root, hi, pool), pool) {
            // αx + β = (−α)(b − x): needs α < 0.
            if sign_of(alpha, pool) != Some(Sign::Negative) {
                return None;
            }
            coeff.push(powr(simp(neg(alpha, pool), pool), e, pool));
            q_parts.push(e);
        } else {
            return None;
        }
    }
    if p_parts.is_empty() || q_parts.is_empty() {
        return None;
    }
    let p = simp(mk_add(p_parts, pool), pool);
    let q = simp(mk_add(q_parts, pool), pool);
    if is_nonneg_int(p, pool) && is_nonneg_int(q, pool) {
        return None; // a polynomial: the FTC route integrates it exactly
    }
    let one = pool.integer(1_i32);
    let a1 = pool.add(vec![p, one]);
    let b1 = pool.add(vec![q, one]);
    let conds = match conditions(&[a1, b1], pool) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    let width = powr(h, pool.add(vec![p, q, one]), pool);
    coeff.push(width);
    coeff.push(beta_closed(a1, b1, pool));
    Some(finish(
        "beta_function_integral",
        expr,
        mk_mul(coeff, pool),
        conds,
        pool,
    ))
}

// ---------------------------------------------------------------------------
// Beta on the half line
// ---------------------------------------------------------------------------

/// `∫₀^∞ c·x^p (γ + δx)^{−r} dx = c·γ^{p+1−r} δ^{−p−1} B(p+1, r−p−1)`.
fn beta_half_line(expr: ExprId, var: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> Outcome {
    if !is_zero(lo, pool) || !is_pos_inf(hi, pool) {
        return None;
    }
    let (mut coeff, powers) = power_factors(expr, var, pool)?;
    let mut p_parts = Vec::new();
    let mut lin: Option<(ExprId, ExprId, ExprId)> = None; // (δ, γ, −r)
    for (base, e) in powers {
        if base == var {
            p_parts.push(e);
            continue;
        }
        let (d, c) = linear(base, var, pool)?;
        if lin.is_some() || is_zero(c, pool) {
            return None;
        }
        if sign_of(d, pool) != Some(Sign::Positive) || sign_of(c, pool) != Some(Sign::Positive) {
            return None;
        }
        lin = Some((d, c, e));
    }
    let (d, c, e) = lin?;
    let p = simp(mk_add(p_parts, pool), pool);
    let r = simp(neg(e, pool), pool);
    if is_integer_literal(p, pool) && is_integer_literal(r, pool) {
        return None; // rational integrand: residue / FTC routes
    }
    let one = pool.integer(1_i32);
    let a1 = simp(pool.add(vec![p, one]), pool);
    let b1 = simp(sub(r, a1, pool), pool);
    let conds = match conditions(&[a1, b1], pool) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    coeff.push(powr(c, simp(sub(a1, r, pool), pool), pool));
    coeff.push(powr(d, simp(neg(a1, pool), pool), pool));
    coeff.push(beta_closed(a1, b1, pool));
    Some(finish(
        "beta_function_integral",
        expr,
        mk_mul(coeff, pool),
        conds,
        pool,
    ))
}

// ---------------------------------------------------------------------------
// Wallis / Beta on [0, π/2]
// ---------------------------------------------------------------------------

/// `∫₀^{π/2} c·sin^p x·cos^q x dx = c·½·B((p+1)/2, (q+1)/2)`.
fn beta_trig(expr: ExprId, var: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> Outcome {
    if !is_zero(lo, pool) {
        return None;
    }
    let half_pi = pool.mul(vec![pool.rational(1, 2), pi(pool)]);
    if !finite_bound(hi, var, pool) || !is_zero(sub(hi, half_pi, pool), pool) {
        return None;
    }
    let (mut coeff, powers) = power_factors(expr, var, pool)?;
    let (mut p_parts, mut q_parts) = (Vec::new(), Vec::new());
    for (base, e) in powers {
        match pool.get(base) {
            ExprData::Func { name, args } if args.len() == 1 && args[0] == var => {
                match name.as_str() {
                    "sin" => p_parts.push(e),
                    "cos" => q_parts.push(e),
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    if p_parts.is_empty() && q_parts.is_empty() {
        return None;
    }
    let p = simp(mk_add(p_parts, pool), pool);
    let q = simp(mk_add(q_parts, pool), pool);
    if is_integer_literal(p, pool) && is_integer_literal(q, pool) {
        return None; // trigonometric polynomial / rational: the FTC route
    }
    let one = pool.integer(1_i32);
    let a1 = pool.add(vec![p, one]);
    let b1 = pool.add(vec![q, one]);
    let conds = match conditions(&[a1, b1], pool) {
        Ok(c) => c,
        Err(e) => return Some(Err(e)),
    };
    let half = pool.rational(1, 2);
    coeff.push(half);
    coeff.push(beta_closed(
        pool.mul(vec![half, a1]),
        pool.mul(vec![half, b1]),
        pool,
    ));
    Some(finish(
        "beta_function_integral",
        expr,
        mk_mul(coeff, pool),
        conds,
        pool,
    ))
}

// ---------------------------------------------------------------------------
// Incomplete gamma
// ---------------------------------------------------------------------------

/// One summand `c·x^p·e^{−r x^k + β}` read off a product.
struct GammaTerm {
    coeff: Vec<ExprId>,
    p: ExprId,
    k: i64,
    rate: ExprId,
}

fn gamma_term(t: ExprId, var: ExprId, pool: &ExprPool) -> Option<GammaTerm> {
    let factors = match pool.get(t) {
        ExprData::Mul(a) => a,
        _ => vec![t],
    };
    let mut coeff = Vec::new();
    let mut p_parts = Vec::new();
    let mut expo: Option<(i64, ExprId)> = None;
    for f in factors {
        if is_free_of(f, var, pool) {
            coeff.push(f);
            continue;
        }
        if f == var {
            p_parts.push(pool.integer(1_i32));
            continue;
        }
        match pool.get(f) {
            ExprData::Pow { base, exp } if base == var && is_free_of(exp, var, pool) => {
                p_parts.push(exp);
            }
            ExprData::Func { name, args }
                if name == "sqrt" && args.len() == 1 && args[0] == var =>
            {
                p_parts.push(pool.rational(1, 2));
            }
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 && expo.is_none() => {
                let c = poly_coeffs(args[0], var, pool)?;
                let zero = pool.integer(0_i32);
                let (k, a) = match c.len() {
                    2 => (1, c[1]),
                    3 if c[1] == zero => (2, c[2]),
                    _ => return None,
                };
                if c[0] != zero {
                    coeff.push(pool.func("exp", vec![c[0]]));
                }
                expo = Some((k, a));
            }
            _ => return None,
        }
    }
    let (k, a) = expo?;
    let rate = simp(neg(a, pool), pool);
    if sign_of(rate, pool) != Some(Sign::Positive) {
        return None;
    }
    Some(GammaTerm {
        coeff,
        p: simp(mk_add(p_parts, pool), pool),
        k,
        rate,
    })
}

/// `∫_a^b Σ cᵢ xᵖⁱ e^{−rᵢ x^{kᵢ}} dx` over `0 ≤ a < b ≤ ∞` by the
/// substitution `u = r xᵏ`: `(1/k) r^{−s} (Γ(s, r aᵏ) − Γ(s, r bᵏ))`,
/// `s = (p+1)/k`.
fn incomplete_gamma(expr: ExprId, var: ExprId, lo: ExprId, hi: ExprId, pool: &ExprPool) -> Outcome {
    if !finite_bound(lo, var, pool) {
        return None;
    }
    match sign_of(lo, pool) {
        Some(Sign::Zero) | Some(Sign::Positive) => {}
        _ => return None,
    }
    let lo_zero = is_zero(lo, pool);
    let hi_inf = is_pos_inf(hi, pool);
    if !hi_inf {
        if !finite_bound(hi, var, pool) {
            return None;
        }
        if sign_of(simp(sub(hi, lo, pool), pool), pool) != Some(Sign::Positive) {
            return None;
        }
    }
    let summands = match pool.get(expr) {
        ExprData::Add(a) => a,
        _ => vec![expr],
    };
    let mut terms = Vec::with_capacity(summands.len());
    for t in summands {
        terms.push(gamma_term(t, var, pool)?);
    }
    // Elementary cases (integer `p`, `k = 1`) are the FTC route's.
    if terms.iter().all(|t| t.k == 1 && is_nonneg_int(t.p, pool)) {
        return None;
    }
    let mut conds = Vec::new();
    let mut parts = Vec::new();
    for t in terms {
        let kk = pool.integer(t.k);
        let s = simp(
            pool.mul(vec![
                pool.add(vec![t.p, pool.integer(1_i32)]),
                pool.pow(kk, pool.integer(-1_i32)),
            ]),
            pool,
        );
        if lo_zero {
            match positive_condition(s, pool) {
                Ok(Some(c)) => conds.push(c),
                Ok(None) => {}
                Err(bad) => return Some(Err(divergent(bad, pool))),
            }
        }
        let at = |b: ExprId| simp(pool.mul(vec![t.rate, pool.pow(b, kk)]), pool);
        let g = match (lo_zero, hi_inf) {
            (true, true) => gamma_of(s, pool),
            (false, true) => pool.func("uppergamma", vec![s, at(lo)]),
            (true, false) => pool.func("lowergamma", vec![s, at(hi)]),
            (false, false) => sub(
                pool.func("uppergamma", vec![s, at(lo)]),
                pool.func("uppergamma", vec![s, at(hi)]),
                pool,
            ),
        };
        let mut f = t.coeff;
        if t.k != 1 {
            f.push(pool.rational(1, t.k));
        }
        f.push(powr(t.rate, neg(s, pool), pool));
        f.push(g);
        parts.push(mk_mul(f, pool));
    }
    Some(finish(
        "incomplete_gamma_integral",
        expr,
        mk_add(parts, pool),
        conds,
        pool,
    ))
}

// ---------------------------------------------------------------------------
// Display: exponents of definite results
// ---------------------------------------------------------------------------

/// Most summands [`distribute`] produces before it gives up.
const MAX_DISTRIBUTED_TERMS: usize = 64;

/// Distribute products over sums (and expand small integer powers of sums) —
/// the polynomial expansion of an exponent, as a list of summands.
fn distribute(e: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    match pool.get(e) {
        ExprData::Add(args) => {
            let mut out = Vec::new();
            for a in args {
                out.extend(distribute(a, pool)?);
                if out.len() > MAX_DISTRIBUTED_TERMS {
                    return None;
                }
            }
            Some(out)
        }
        ExprData::Mul(args) => {
            let mut acc: Vec<Vec<ExprId>> = vec![Vec::new()];
            for a in args {
                let parts = distribute(a, pool)?;
                if acc.len() * parts.len() > MAX_DISTRIBUTED_TERMS {
                    return None;
                }
                let mut next = Vec::with_capacity(acc.len() * parts.len());
                for prefix in &acc {
                    for &p in &parts {
                        let mut t = prefix.clone();
                        t.push(p);
                        next.push(t);
                    }
                }
                acc = next;
            }
            Some(acc.into_iter().map(|f| mk_mul(f, pool)).collect())
        }
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Integer(n) if matches!(pool.get(base), ExprData::Add(_)) => {
                let k = n.0.to_i64().filter(|k| (2..=4).contains(k))?;
                let parts = distribute(base, pool)?;
                let mut acc = parts.clone();
                for _ in 1..k {
                    if acc.len() * parts.len() > MAX_DISTRIBUTED_TERMS {
                        return None;
                    }
                    let mut next = Vec::with_capacity(acc.len() * parts.len());
                    for &a in &acc {
                        for &b in &parts {
                            next.push(pool.mul(vec![a, b]));
                        }
                    }
                    acc = next;
                }
                Some(acc)
            }
            _ => Some(vec![e]),
        },
        _ => Some(vec![e]),
    }
}

fn size(e: ExprId, pool: &ExprPool) -> usize {
    1 + match pool.get(e) {
        ExprData::Add(a) | ExprData::Mul(a) | ExprData::Func { args: a, .. } => {
            a.iter().map(|&x| size(x, pool)).sum()
        }
        ExprData::Pow { base, exp } => size(base, pool) + size(exp, pool),
        _ => 0,
    }
}

/// Rewrite the argument of every `exp` in a definite result to its expanded,
/// collected form when that is smaller: the FTC difference of the
/// convolution `∫₀ᵗ e^{−aτ}e^{−b(t−τ)} dτ` substitutes `τ = t` into
/// `e^{(b−a)τ − bt}` and leaves `e^{−bt + t(b−a)}`, which is `e^{−at}`.
/// Only the spelling changes (expansion and collection are identities).
pub(crate) fn tidy_exponents(e: ExprId, pool: &ExprPool) -> ExprId {
    fn go(e: ExprId, pool: &ExprPool, changed: &mut bool) -> ExprId {
        match pool.get(e) {
            ExprData::Func { name, args } => {
                let new_args: Vec<ExprId> = args.iter().map(|&a| go(a, pool, changed)).collect();
                if name == "exp" && new_args.len() == 1 {
                    let arg = new_args[0];
                    if let Some(terms) = distribute(arg, pool) {
                        let cand = simp(mk_add(terms, pool), pool);
                        if size(cand, pool) < size(arg, pool) {
                            *changed = true;
                            return pool.func("exp", vec![cand]);
                        }
                    }
                }
                if new_args != args {
                    pool.func(&name, new_args)
                } else {
                    e
                }
            }
            ExprData::Add(args) => {
                let n: Vec<ExprId> = args.iter().map(|&a| go(a, pool, changed)).collect();
                if n != args {
                    pool.add(n)
                } else {
                    e
                }
            }
            ExprData::Mul(args) => {
                let n: Vec<ExprId> = args.iter().map(|&a| go(a, pool, changed)).collect();
                if n != args {
                    pool.mul(n)
                } else {
                    e
                }
            }
            ExprData::Pow { base, exp } => {
                let (b, x) = (go(base, pool, changed), go(exp, pool, changed));
                if (b, x) != (base, exp) {
                    pool.pow(b, x)
                } else {
                    e
                }
            }
            _ => e,
        }
    }
    let mut changed = false;
    let out = go(e, pool, &mut changed);
    if changed {
        simp(out, pool)
    } else {
        e
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn val(e: ExprId, env: &[(ExprId, f64)], pool: &ExprPool) -> f64 {
        let m: HashMap<ExprId, f64> = env.iter().copied().collect();
        crate::jit::eval_interp(e, &m, pool).expect("evaluates")
    }

    #[test]
    fn beta_with_symbolic_exponents_states_conditions() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let a = p.symbol("a", Domain::Real);
        let b = p.symbol("b", Domain::Real);
        let one = p.integer(1_i32);
        let f = p.mul(vec![p.pow(x, a), p.pow(p.add(vec![one, neg(x, &p)]), b)]);
        let r = try_definite(f, x, p.integer(0_i32), one, &p)
            .unwrap()
            .unwrap();
        let conds: Vec<_> = r
            .log
            .0
            .iter()
            .flat_map(|s| s.side_conditions.clone())
            .collect();
        assert_eq!(conds.len(), 2, "a > −1 and b > −1 must be stated");
        // mpmath.beta(2.3, 1.4)
        let exact = 0.248_206_110_842_000_3;
        let got = val(r.value, &[(a, 1.3), (b, 0.4)], &p);
        assert!((got - exact).abs() < 1e-12, "{got} vs {exact}");
    }

    #[test]
    fn beta_divergent_exponent_is_refused() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let one = p.integer(1_i32);
        let f = p.mul(vec![
            p.pow(x, p.rational(-3, 2)),
            p.pow(p.add(vec![one, neg(x, &p)]), p.rational(1, 2)),
        ]);
        let r = try_definite(f, x, p.integer(0_i32), one, &p).unwrap();
        assert!(r.is_err());
    }

    #[test]
    fn upper_incomplete_gamma_from_nonzero_bound() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let s = p.symbol("s", Domain::Positive);
        let f = p.mul(vec![
            p.pow(x, p.add(vec![s, p.integer(-1_i32)])),
            p.func("exp", vec![neg(x, &p)]),
        ]);
        let r = try_definite(f, x, p.integer(2_i32), p.pos_infinity(), &p)
            .unwrap()
            .unwrap();
        assert!(p.display(r.value).to_string().contains("uppergamma"));
    }
}
