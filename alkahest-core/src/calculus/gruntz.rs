//! Gruntz comparability-graph algorithm for symbolic limits of exp-log combinations (V2-17).
//!
//! Reference: Gruntz (1996) "On Computing Limits in a Symbolic Manipulation System"
//!
//! # Algorithm (limit as var → +∞)
//!
//! 1. Collect all `exp(h)` subexpressions where `h` diverges to ±∞.
//! 2. Build the comparability ordering: `exp(h₁) ≻ exp(h₂)` iff `lim(h₁/h₂) = ∞`.
//! 3. Retain only the **maximal** (MRV) set — the fastest-growing elements.
//! 4. Pick ω ∈ MRV with ω → 0⁺ (inner h_ω → -∞).  If all inners → +∞, form ω = exp(-h).
//! 5. Rewrite expr: each element `exp(hₑ)` ∈ MRV becomes `ω^c`, c = lim(hₑ/h_ω).
//! 6. Expand the rewritten expression as a Laurent series in ω → 0⁺.
//! 7. Leading power > 0 → 0; < 0 → ±∞; = 0 → recurse on the coefficient.

use crate::calculus::limits::{limit, LimitDirection, LimitError};
use crate::calculus::series::local_expansion;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::rules::{CollectExp, ExpPow};
use crate::simplify::{rules_for_config, simplify, simplify_with, SimplifyConfig};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Recursion guard — prevents unbounded re-entry when Gruntz sub-limits
// trigger Gruntz again on simpler expressions.
// ---------------------------------------------------------------------------

thread_local! {
    static GRUNTZ_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

const GRUNTZ_MAX_DEPTH: u32 = 8;
const SERIES_TERMS: u32 = 8;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Try to compute `lim_{var → +∞} expr` using the Gruntz algorithm.
/// Returns `None` if not applicable or if a sub-step fails.
pub(crate) fn try_gruntz(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<ExprId>, LimitError> {
    let current = GRUNTZ_DEPTH.with(|d| d.get());
    if current >= GRUNTZ_MAX_DEPTH {
        return Ok(None);
    }
    GRUNTZ_DEPTH.with(|d| d.set(current + 1));
    let result = gruntz_inner(expr, var, pool);
    GRUNTZ_DEPTH.with(|d| d.set(current));
    result
}

/// Collapse exp(h)^n → exp(n·h) and exp(a)·exp(b) → exp(a+b) before analysis.
/// These rules must NOT be in the default simplifier (they undo omega-power rewrites),
/// so we apply them here only as an input preprocessing step, combined with the full
/// default rule set so that exponent arithmetic is fully simplified.
fn preprocess_exp(expr: ExprId, pool: &ExprPool) -> ExprId {
    let cfg = SimplifyConfig::default();
    let mut rules = rules_for_config(&cfg);
    rules.push(Box::new(ExpPow));
    rules.push(Box::new(CollectExp));
    simplify_with(expr, pool, &rules, cfg).value
}

fn gruntz_inner(expr: ExprId, var: ExprId, pool: &ExprPool) -> Result<Option<ExprId>, LimitError> {
    // Collapse exp(h)^n → exp(n·h) and merge exp factors before analysis.
    let preprocessed = preprocess_exp(expr, pool);

    // Collect exp(h) subexpressions where h diverges
    let mut candidates: Vec<(ExprId, ExprId)> = Vec::new();
    collect_exp_subexprs(preprocessed, var, pool, &mut candidates)?;

    if candidates.is_empty() {
        // If preprocessing changed the expression (e.g. exp(x+1)/exp(x) → exp(1)),
        // delegate to limit() on the simplified form so it can use direct substitution.
        if preprocessed != expr {
            let lim = limit(
                preprocessed,
                var,
                pool.pos_infinity(),
                LimitDirection::Bidirectional,
                pool,
            )?;
            return Ok(Some(lim));
        }
        return Ok(None);
    }
    let expr = preprocessed;

    // Build MRV set: keep only the maximally-growing elements
    let mrv = build_mrv_set(candidates, var, pool)?;
    if mrv.is_empty() {
        return Ok(None);
    }

    // Find ω: the mrv element that → 0, or construct one
    let (omega_expr, omega_inner) = match find_omega(&mrv, var, pool)? {
        Some(p) => p,
        None => return Ok(None),
    };

    // Rewrite expr by substituting mrv elements as powers of ω
    let rewritten = match rewrite_in_omega(expr, &mrv, omega_expr, omega_inner, var, pool) {
        Ok(r) => r,
        Err(_) => return Ok(None),
    };

    // Expand as Laurent series in ω → 0⁺ and find the leading term
    let (coeff, power) = match leading_term_at_zero(rewritten, omega_expr, pool)? {
        Some(lt) => lt,
        None => return Ok(None),
    };

    if power > 0 {
        return Ok(Some(pool.integer(0_i32)));
    }
    if power < 0 {
        let sign = sign_of_coeff_at_inf(coeff, var, pool);
        return Ok(Some(signed_infinity(pool, sign)));
    }

    // power == 0: the limit equals lim_{x→+∞} coeff(x)
    let lim_coeff = limit(
        coeff,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    )?;
    Ok(Some(lim_coeff))
}

// ---------------------------------------------------------------------------
// Step 1 — collect diverging exp subexpressions
// ---------------------------------------------------------------------------

/// Walk `expr` and collect `(exp(h), h)` pairs where `h` contains `var` and
/// `limit(h, var, +∞)` is ±∞.
fn collect_exp_subexprs(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
    out: &mut Vec<(ExprId, ExprId)>,
) -> Result<(), LimitError> {
    match pool.get(expr) {
        ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
            let inner = args[0];
            if contains_var(inner, var, pool) {
                let lim = limit(
                    inner,
                    var,
                    pool.pos_infinity(),
                    LimitDirection::Bidirectional,
                    pool,
                );
                if let Ok(l) = lim {
                    if is_pos_inf(l, pool) || is_neg_inf(l, pool) {
                        out.push((expr, inner));
                        // Recurse into inner to discover nested exp terms
                        collect_exp_subexprs(inner, var, pool, out)?;
                        return Ok(());
                    }
                }
            }
            // Inner doesn't diverge — still recurse in case args have nested exp
            for a in args.iter() {
                collect_exp_subexprs(*a, var, pool, out)?;
            }
        }
        ExprData::Add(xs) | ExprData::Mul(xs) => {
            for x in xs {
                collect_exp_subexprs(x, var, pool, out)?;
            }
        }
        ExprData::Pow { base, exp } => {
            collect_exp_subexprs(base, var, pool, out)?;
            collect_exp_subexprs(exp, var, pool, out)?;
        }
        ExprData::Func { args, .. } => {
            for a in args {
                collect_exp_subexprs(a, var, pool, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 2 — comparability and MRV set construction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Growth {
    Slower,
    Equal,
    Faster,
}

/// Compare growth rates of `exp(h1)` and `exp(h2)` via `lim(h1/h2, x→+∞)`.
fn compare_growth(
    h1: ExprId,
    h2: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Growth, LimitError> {
    if h1 == h2 {
        return Ok(Growth::Equal);
    }
    let ratio = simplify(pool.mul(vec![h1, pool.pow(h2, pool.integer(-1_i32))]), pool).value;
    let l = limit(
        ratio,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    )?;
    if is_zero(l, pool) {
        Ok(Growth::Slower)
    } else if is_pos_inf(l, pool) || is_neg_inf(l, pool) {
        Ok(Growth::Faster)
    } else {
        Ok(Growth::Equal)
    }
}

/// Retain only elements of `candidates` that are not dominated by any other.
fn build_mrv_set(
    candidates: Vec<(ExprId, ExprId)>,
    var: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, ExprId)>, LimitError> {
    // De-duplicate by exp ExprId
    let mut unique: Vec<(ExprId, ExprId)> = Vec::new();
    for c in candidates {
        if !unique.iter().any(|(e, _)| *e == c.0) {
            unique.push(c);
        }
    }
    if unique.len() == 1 {
        return Ok(unique);
    }

    let n = unique.len();
    let mut dominated = vec![false; n];
    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[i] {
                continue;
            }
            let h_i = unique[i].1;
            let h_j = unique[j].1;
            if let Ok(Growth::Slower) = compare_growth(h_i, h_j, var, pool) {
                dominated[i] = true;
            }
        }
    }
    Ok(unique
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !dominated[*i])
        .map(|(_, v)| v)
        .collect())
}

// ---------------------------------------------------------------------------
// Step 3 — find ω
// ---------------------------------------------------------------------------

/// Return `(omega_expr, omega_inner)` where `omega_expr → 0⁺`.
///
/// Prefers an mrv element whose inner → -∞.  If none exists, constructs
/// `exp(-h)` from the first element whose inner → +∞.
fn find_omega(
    mrv: &[(ExprId, ExprId)],
    var: ExprId,
    pool: &ExprPool,
) -> Result<Option<(ExprId, ExprId)>, LimitError> {
    for &(e, h) in mrv {
        let lim = limit(
            h,
            var,
            pool.pos_infinity(),
            LimitDirection::Bidirectional,
            pool,
        )?;
        if is_neg_inf(lim, pool) {
            return Ok(Some((e, h)));
        }
    }
    // All inners → +∞: build ω = exp(-h) from the first
    if let Some(&(_, h)) = mrv.first() {
        let neg_h = simplify(pool.mul(vec![pool.integer(-1_i32), h]), pool).value;
        let omega = simplify(pool.func("exp".to_string(), vec![neg_h]), pool).value;
        return Ok(Some((omega, neg_h)));
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Step 4 — rewrite in ω
// ---------------------------------------------------------------------------

/// Build the rewritten expression: every mrv `exp(hₑ)` → `ω^c`, c = lim(hₑ/h_ω).
fn rewrite_in_omega(
    expr: ExprId,
    mrv: &[(ExprId, ExprId)],
    omega_expr: ExprId,
    omega_inner: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    let mut subst: HashMap<ExprId, ExprId> = HashMap::new();
    for &(e, h_e) in mrv {
        if e == omega_expr {
            subst.insert(e, omega_expr);
            continue;
        }
        let c = exponent_relative_to_omega(h_e, omega_inner, var, pool)?;
        let rep = simplify(pool.pow(omega_expr, c), pool).value;
        subst.insert(e, rep);
    }
    Ok(rewrite_node(expr, &subst, pool))
}

fn exponent_relative_to_omega(
    h_e: ExprId,
    h_omega: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, LimitError> {
    if h_e == h_omega {
        return Ok(pool.integer(1_i32));
    }
    let ratio = simplify(
        pool.mul(vec![h_e, pool.pow(h_omega, pool.integer(-1_i32))]),
        pool,
    )
    .value;
    limit(
        ratio,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    )
}

fn rewrite_node(expr: ExprId, subst: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    if let Some(&rep) = subst.get(&expr) {
        return rep;
    }
    match pool.get(expr) {
        ExprData::Add(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| rewrite_node(*x, subst, pool)).collect();
            simplify(pool.add(ys), pool).value
        }
        ExprData::Mul(xs) => {
            let ys: Vec<ExprId> = xs.iter().map(|x| rewrite_node(*x, subst, pool)).collect();
            simplify(pool.mul(ys), pool).value
        }
        ExprData::Pow { base, exp } => {
            let b = rewrite_node(base, subst, pool);
            let e = rewrite_node(exp, subst, pool);
            // Flatten (x^a)^b → x^(a·b) so rational omega powers compose correctly.
            if let ExprData::Pow {
                base: inner_base,
                exp: inner_e,
            } = pool.get(b)
            {
                let combined = simplify(pool.mul(vec![inner_e, e]), pool).value;
                return simplify(pool.pow(inner_base, combined), pool).value;
            }
            simplify(pool.pow(b, e), pool).value
        }
        ExprData::Func { name, args } => {
            let na: Vec<ExprId> = args.iter().map(|a| rewrite_node(*a, subst, pool)).collect();
            simplify(pool.func(name.clone(), na), pool).value
        }
        _ => expr,
    }
}

// ---------------------------------------------------------------------------
// Step 5 — leading term of the rewritten expression as ω → 0⁺
// ---------------------------------------------------------------------------

fn leading_term_at_zero(
    expr: ExprId,
    omega: ExprId,
    pool: &ExprPool,
) -> Result<Option<(ExprId, i32)>, LimitError> {
    // Fast path: symbolically factor out the omega power.
    // This handles rational exponents and avoids Taylor-fallback singularity issues.
    if let Some((coeff, rat_power)) = factor_omega_power(expr, omega, pool) {
        let sentinel: i32 = if rat_power > 0 {
            1
        } else if rat_power < 0 {
            -1
        } else {
            0
        };
        return Ok(Some((coeff, sentinel)));
    }
    // Fallback: full Laurent series expansion (for expressions without a pure omega factor).
    let zero = pool.integer(0_i32);
    let expansion = match local_expansion(expr, omega, zero, SERIES_TERMS, pool) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };
    let base_val = expansion.valuation;
    for (k, coeff) in expansion.coeffs.iter().enumerate() {
        let c = simplify(*coeff, pool).value;
        if !is_zero(c, pool) {
            return Ok(Some((c, base_val + k as i32)));
        }
    }
    Ok(None)
}

/// Factor `expr` as `(coeff, power)` where `expr = coeff · omega^power`
/// and `coeff` is omega-free.  Returns `None` if `expr` contains no omega.
fn factor_omega_power(
    expr: ExprId,
    omega: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, rug::Rational)> {
    if expr == omega {
        return Some((pool.integer(1_i32), rug::Rational::from(1)));
    }
    match pool.get(expr) {
        ExprData::Pow { base, exp } => {
            if base == omega {
                let e_rat = expr_as_rational(exp, pool)?;
                return Some((pool.integer(1_i32), e_rat));
            }
            // (base^a)^b → base^(a·b)
            let (inner_coeff, inner_p) = factor_omega_power(base, omega, pool)?;
            let b_rat = expr_as_rational(exp, pool)?;
            let combined = inner_p * b_rat.clone();
            let new_coeff = simplify(pool.pow(inner_coeff, exp), pool).value;
            Some((new_coeff, combined))
        }
        ExprData::Mul(xs) => {
            let mut total_p = rug::Rational::from(0);
            let mut coeff_factors: Vec<ExprId> = Vec::new();
            let mut found = false;
            for &x in &xs {
                if let Some((c, p)) = factor_omega_power(x, omega, pool) {
                    found = true;
                    total_p += p;
                    if !matches!(pool.get(c), ExprData::Integer(n) if n.0 == 1) {
                        coeff_factors.push(c);
                    }
                } else {
                    coeff_factors.push(x);
                }
            }
            if !found {
                return None;
            }
            let coeff = match coeff_factors.len() {
                0 => pool.integer(1_i32),
                1 => coeff_factors[0],
                _ => simplify(pool.mul(coeff_factors), pool).value,
            };
            Some((coeff, total_p))
        }
        _ => None,
    }
}

fn expr_as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_var(expr: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if expr == var {
        return true;
    }
    match pool.get(expr) {
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().any(|a| contains_var(*a, var, pool)),
        ExprData::Pow { base, exp } => {
            contains_var(base, var, pool) || contains_var(exp, var, pool)
        }
        ExprData::Func { args, .. } => args.iter().any(|a| contains_var(*a, var, pool)),
        _ => false,
    }
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    matches!(pool.get(e), ExprData::Integer(n) if n.0 == 0)
        || matches!(pool.get(e), ExprData::Rational(r) if r.0 == 0)
}

fn is_pos_inf(e: ExprId, pool: &ExprPool) -> bool {
    use crate::kernel::pool::POS_INFINITY_SYMBOL;
    matches!(pool.get(e), ExprData::Symbol { name, .. } if name == POS_INFINITY_SYMBOL)
}

fn is_neg_inf(e: ExprId, pool: &ExprPool) -> bool {
    let ExprData::Mul(args) = pool.get(e) else {
        return false;
    };
    let m1 = pool.integer(-1_i32);
    args.len() == 2
        && ((args[0] == m1 && is_pos_inf(args[1], pool))
            || (args[1] == m1 && is_pos_inf(args[0], pool)))
}

fn signed_infinity(pool: &ExprPool, sign: i8) -> ExprId {
    if sign < 0 {
        pool.mul(vec![pool.integer(-1_i32), pool.pos_infinity()])
    } else {
        pool.pos_infinity()
    }
}

/// Determine the sign of `coeff` as `var → +∞` (for the leading-term decision).
fn sign_of_coeff_at_inf(coeff: ExprId, var: ExprId, pool: &ExprPool) -> i8 {
    // Try computing lim(coeff) and reading its sign
    if let Ok(l) = limit(
        coeff,
        var,
        pool.pos_infinity(),
        LimitDirection::Bidirectional,
        pool,
    ) {
        if let Some(s) = structural_sign(l, pool) {
            return s;
        }
    }
    structural_sign(coeff, pool).unwrap_or(1)
}

fn structural_sign(e: ExprId, pool: &ExprPool) -> Option<i8> {
    match pool.get(e) {
        ExprData::Integer(n) => {
            if n.0 > 0 {
                Some(1)
            } else if n.0 < 0 {
                Some(-1)
            } else {
                None
            }
        }
        ExprData::Rational(r) => {
            if r.0 == 0 {
                None
            } else if r.0 > 0 {
                Some(1)
            } else {
                Some(-1)
            }
        }
        ExprData::Mul(xs) => {
            let mut s = 1i8;
            for x in xs {
                s *= structural_sign(x, pool)?;
            }
            Some(s)
        }
        ExprData::Pow { exp, .. } if matches!(pool.get(exp), ExprData::Integer(n) if n.0.clone() % 2 == 0) => {
            Some(1)
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculus::limits::{limit, LimitDirection};
    use crate::kernel::Domain;

    fn oo(pool: &ExprPool) -> ExprId {
        pool.pos_infinity()
    }

    #[test]
    fn gruntz_exp_neg_x_to_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(-x) = 0
        let expr = p.func("exp".to_string(), vec![p.mul(vec![p.integer(-1_i32), x])]);
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "exp(-x): {}", p.display(r));
    }

    #[test]
    fn gruntz_x_times_exp_neg_x() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} x * exp(-x) = 0
        let neg_x = p.mul(vec![p.integer(-1_i32), x]);
        let expr = simplify(p.mul(vec![x, p.func("exp".to_string(), vec![neg_x])]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "x*exp(-x): {}", p.display(r));
    }

    #[test]
    fn gruntz_exp_x_over_x_squared_is_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(x) / x² = +∞
        let expr = simplify(
            p.mul(vec![
                p.func("exp".to_string(), vec![x]),
                p.pow(x, p.integer(-2_i32)),
            ]),
            &p,
        )
        .value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, oo(&p), "exp(x)/x²: {}", p.display(r));
    }

    #[test]
    fn gruntz_ratio_exp2x_over_exp3x_is_zero() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(2x) / exp(3x) = 0
        let e2x = p.func("exp".to_string(), vec![p.mul(vec![p.integer(2_i32), x])]);
        let e3x = p.func("exp".to_string(), vec![p.mul(vec![p.integer(3_i32), x])]);
        let expr = simplify(p.mul(vec![e2x, p.pow(e3x, p.integer(-1_i32))]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, p.integer(0_i32), "exp(2x)/exp(3x): {}", p.display(r));
    }

    #[test]
    fn gruntz_nested_exp_ratio() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(x+1) / exp(x) = exp(1)
        let e_x1 = p.func("exp".to_string(), vec![p.add(vec![x, p.integer(1_i32)])]);
        let e_x = p.func("exp".to_string(), vec![x]);
        let expr = simplify(p.mul(vec![e_x1, p.pow(e_x, p.integer(-1_i32))]), &p).value;
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        let expected = simplify(p.func("exp".to_string(), vec![p.integer(1_i32)]), &p).value;
        let r_s = simplify(r, &p).value;
        assert_eq!(r_s, expected, "exp(x+1)/exp(x): {}", p.display(r));
    }

    #[test]
    fn gruntz_nested_exp_exp_x_is_inf() {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        // lim_{x→+∞} exp(exp(x)) = +∞
        let inner = p.func("exp".to_string(), vec![x]);
        let expr = p.func("exp".to_string(), vec![inner]);
        let r = limit(expr, x, oo(&p), LimitDirection::Bidirectional, &p).unwrap();
        assert_eq!(r, oo(&p), "exp(exp(x)): {}", p.display(r));
    }
}
