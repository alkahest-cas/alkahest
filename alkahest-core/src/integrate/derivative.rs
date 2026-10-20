//! Integration of integrands that contain **undefined functions** and their
//! `Derivative` nodes (`f(t)`, `Derivative(f(t), t)`, …).
//!
//! An undefined function has no antiderivative of its own, so the only
//! integrals with a closed form are the ones that are *exact derivatives*:
//!
//! 1. **Chain-rule reversal.**  For a candidate `w` — an undefined-function
//!    node `f⁽ᵏ⁾(g(t))` occurring in the term, or one derivative order
//!    below such a node — the term `T` is `H(w)·w′` when `T / w′` becomes free
//!    of `t` and of every undefined function once `w` is replaced by a fresh
//!    symbol `s`.  Then `∫T dt = (∫H ds)|_{s=w}`.  This covers
//!    `∫ f′ dt = f`, `∫ f f′ dt = f²/2`, `∫ f′/f dt = log f`,
//!    `∫ f′ eᶠ dt = eᶠ`, `∫ f″ f′ dt = f′²/2`, and `∫ 2t f′(t²) dt = f(t²)`.
//! 2. **Polynomial × derivative.**  `∫ p(t)·f⁽ᵏ⁾(t) dt` by parts `k` times,
//!    closed exactly when `deg p < k` (`∫ t f″ dt = t f′ − f`).
//!
//! Every candidate antiderivative is differentiated back and compared with the
//! integrand (`simplify(F′ − T) = 0`) before it is returned.  Anything else —
//! `∫ f(t) dt`, `∫ t f(t) dt` — has no closed form in terms of `f` and is a
//! coded refusal; no unevaluated-integral node is produced.  The integrals are
//! formal: they presume `f` is smooth enough for the derivatives that appear.

use super::engine::{integrate, IntegrationError};
use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::diff::diff;
use crate::diff::undefined::{var_dependent_nodes, view};
use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::simplify_expanded;
use std::collections::HashMap;

type IntResult = Result<DerivedExpr<ExprId>, IntegrationError>;

fn refuse<T>(msg: impl Into<String>) -> Result<T, IntegrationError> {
    Err(IntegrationError::NotImplemented(msg.into()))
}

/// `Some(result)` when `expr` has an undefined function (or a derivative of
/// one) of `var`.
pub(crate) fn try_integrate(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<IntResult> {
    if var_dependent_nodes(expr, var, pool).is_empty() {
        return None;
    }
    Some(integrate_undefined(expr, var, pool))
}

/// Definite integral `∫_lower^upper expr d var` of an integrand with an
/// undefined function: `F(upper) − F(lower)` of the exact antiderivative.
/// Infinite bounds are refused (nothing is known about `f` there).
pub(crate) fn try_definite(
    expr: ExprId,
    var: ExprId,
    lower: ExprId,
    upper: ExprId,
    pool: &ExprPool,
) -> Option<IntResult> {
    if var_dependent_nodes(expr, var, pool).is_empty() {
        return None;
    }
    Some((|| {
        if crate::kernel::contains_non_finite_atom(pool, lower)
            || crate::kernel::contains_non_finite_atom(pool, upper)
        {
            return refuse(
                "an improper integral of an undefined function has no value: nothing is known \
                 about the function at infinity",
            );
        }
        let f = integrate_undefined(expr, var, pool)?;
        let at = |p: ExprId| {
            let mut m = HashMap::new();
            m.insert(var, p);
            subs(f.value, &m, pool)
        };
        let v = simplify(
            pool.add(vec![at(upper), pool.mul(vec![pool.integer(-1), at(lower)])]),
            pool,
        )
        .value;
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("integrate_derivative_ftc", expr, v));
        Ok(DerivedExpr::with_log(v, f.log.merge(log)))
    })())
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simplify(e, pool).value;
    matches!(pool.get(s), ExprData::Integer(n) if n.0 == 0)
}

/// `d/dvar` simplified.
fn d(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    diff(e, var, pool)
        .ok()
        .map(|r| simplify(r.value, pool).value)
}

fn integrate_undefined(expr: ExprId, var: ExprId, pool: &ExprPool) -> IntResult {
    let e = simplify(expr, pool).value;
    let mut terms = match pool.get(e) {
        ExprData::Add(a) => a,
        _ => vec![e],
    };
    let mut total = Vec::new();
    let mut log = DerivationLog::new();
    // Terms linear in one undefined function (`q(t)·f⁽ʲ⁾(t)`) are integrated
    // together: `t·f′ + f` is the derivative of `t·f` although neither term is.
    let (lin, other): (Vec<ExprId>, Vec<ExprId>) = terms
        .iter()
        .partition(|&&t| is_linear_in_one_node(t, var, pool));
    terms = other;
    if !lin.is_empty() {
        let sum = simplify(pool.add(lin.clone()), pool).value;
        match linear_exact(sum, var, pool).filter(|&f| antiderivative_checks(f, sum, var, pool)) {
            Some(f) => total.push(f),
            None => terms.extend(lin),
        }
    }
    // A first pass keeps the terms as written; a term that no rule closes is
    // expanded and retried, since `f′·(f + 1)` is one chain-rule term but
    // `t·(f′ + f″)` only closes termwise.
    while let Some(term) = terms.pop() {
        match integrate_term(term, var, pool) {
            Ok(f) => total.push(f),
            Err(err) => {
                let ex = simplify_expanded(term, pool).value;
                if ex != term {
                    let parts = match pool.get(ex) {
                        ExprData::Add(a) => a,
                        _ => vec![ex],
                    };
                    if parts.len() > 1 {
                        terms.extend(parts);
                        continue;
                    }
                }
                return Err(err);
            }
        }
    }
    let sum = simplify(pool.add(total), pool).value;
    log.push(RewriteStep::simple("integrate_derivative", expr, sum));
    Ok(DerivedExpr::with_log(sum, log))
}

fn integrate_term(term: ExprId, var: ExprId, pool: &ExprPool) -> Result<ExprId, IntegrationError> {
    // Factors free of `var` (and of undefined functions of it) are constants.
    let factors = match pool.get(term) {
        ExprData::Mul(a) => a,
        _ => vec![term],
    };
    let (mut consts, mut rest) = (Vec::new(), Vec::new());
    for f in factors {
        if mentions_var(f, var, pool) {
            rest.push(f);
        } else {
            consts.push(f);
        }
    }
    let r = pool.mul(rest);
    let c = pool.mul(consts);
    let r = simplify(r, pool).value;
    if var_dependent_nodes(r, var, pool).is_empty() {
        // Not an undefined-function term at all (e.g. `t`): ordinary engine.
        let f = integrate(r, var, pool)?.value;
        return Ok(simplify(pool.mul(vec![c, f]), pool).value);
    }
    let f = chain_rule(r, var, pool).ok_or_else(|| {
        IntegrationError::NotImplemented(format!(
            "integrate: {} has no closed form in terms of its undefined function(s): only \
                 exact derivatives integrate (f′ → f, f·f′ → f²/2, f′/f → log f, \
                 g′(t)·h(g(t)) patterns, and p(t)·f⁽ᵏ⁾(t) with deg p < k). An undefined \
                 function has no antiderivative of its own, so ∫ f dt is left unsolved",
            pool.display(r)
        ))
    })?;
    if !antiderivative_checks(f, r, var, pool) {
        return refuse(format!(
            "the candidate antiderivative {} of {} does not differentiate back to the \
             integrand; declining",
            pool.display(f),
            pool.display(r)
        ));
    }
    Ok(simplify(pool.mul(vec![c, f]), pool).value)
}

/// Candidate inner functions for the chain-rule reversal.
fn candidates(r: ExprId, var: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out: Vec<ExprId> = Vec::new();
    for n in var_dependent_nodes(r, var, pool) {
        if !out.contains(&n) {
            out.push(n);
        }
        if let Some(v) = view(n, pool) {
            for j in 0..v.orders.len() {
                if v.orders[j] > 0 {
                    let mut o = v.orders.clone();
                    o[j] -= 1;
                    let low = crate::diff::undefined::build(v.inner, &o, pool);
                    if !out.contains(&low) {
                        out.push(low);
                    }
                }
            }
        }
    }
    out
}

fn chain_rule(r: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let s = pool.symbol("__alkahest_int_w", Domain::Real);
    for w in candidates(r, var, pool) {
        let dw = d(w, var, pool)?;
        if is_zero(dw, pool) {
            continue;
        }
        let h = simplify(pool.mul(vec![r, pool.pow(dw, pool.integer(-1))]), pool).value;
        let mut m = HashMap::new();
        m.insert(w, s);
        let h = subs(h, &m, pool);
        if mentions_var(h, var, pool) || crate::diff::undefined::has_undefined_function(h, pool) {
            continue;
        }
        let g = integrate(h, s, pool).ok()?.value;
        if mentions_var(g, var, pool) {
            continue;
        }
        let mut back = HashMap::new();
        back.insert(s, w);
        return Some(simplify(subs(g, &back, pool), pool).value);
    }
    None
}

/// `F′ == r`, decided by simplification.
fn antiderivative_checks(f: ExprId, r: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    d(f, var, pool)
        .is_some_and(|b| is_zero(pool.add(vec![b, pool.mul(vec![pool.integer(-1), r])]), pool))
}

/// Is `term` exactly `q·n` for one undefined-function node `n` of the form
/// `f⁽ʲ⁾(var)` and a `q` free of every such node?
fn is_linear_in_one_node(term: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let nodes = var_dependent_nodes(term, var, pool);
    let [n] = nodes.as_slice() else {
        return false;
    };
    if !view(*n, pool).is_some_and(|v| v.args.as_slice() == [var]) {
        return false;
    }
    let s = pool.symbol("__alkahest_int_lin", Domain::Real);
    let mut m = HashMap::new();
    m.insert(*n, s);
    let e = subs(term, &m, pool);
    // `e` must be `q(t)·s` with `q` free of `s` and of other functions.
    let Some(q) = d(e, s, pool) else {
        return false;
    };
    !crate::ode::dsolve::contains(q, s, pool)
        && var_dependent_nodes(q, var, pool).is_empty()
        && is_zero(
            pool.add(vec![e, pool.mul(vec![pool.integer(-1), q, s])]),
            pool,
        )
}

/// Antiderivative of `Σ qⱼ(t)·f⁽ʲ⁾(t)` (one undefined function `f` of `var`,
/// coefficients free of it) when it is again of that form.
///
/// Integrating by parts through every term,
/// `Σ qⱼ f⁽ʲ⁾ = d/dt[Σⱼ Σ_{i<j} (−1)ⁱ qⱼ⁽ⁱ⁾ f⁽ʲ⁻¹⁻ⁱ⁾] + (Σⱼ (−1)ʲ qⱼ⁽ʲ⁾)·f`,
/// so the sum is an exact derivative iff the coefficient of the leftover `f`
/// vanishes identically.  (`∫ t f″ = t f′ − f`; `∫ (t f′ + f) = t f`; but
/// `∫ f` and `∫ t f′ = t f − ∫ f` have none.)
fn linear_exact(sum: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let nodes = var_dependent_nodes(sum, var, pool);
    let mut inner = None;
    let mut by_order: Vec<(u32, ExprId)> = Vec::new();
    for &n in &nodes {
        let v = view(n, pool)?;
        if v.args.as_slice() != [var] {
            return None;
        }
        match inner {
            None => inner = Some(v.inner),
            Some(i) if i != v.inner => return None,
            _ => {}
        }
        by_order.push((v.orders[0], n));
    }
    let inner = inner?;
    let s = pool.symbol("__alkahest_int_lin", Domain::Real);
    let mut terms = Vec::new();
    let mut residual = Vec::new();
    for &(j, n) in &by_order {
        let mut m = HashMap::new();
        m.insert(n, s);
        let e = subs(sum, &m, pool);
        // Coefficient of this node: the other nodes are independent unknowns,
        // so differentiate with the others frozen at 0.
        let mut zero_others = HashMap::new();
        for &(_, o) in &by_order {
            if o != n {
                zero_others.insert(o, pool.integer(0));
            }
        }
        let e = subs(e, &zero_others, pool);
        let q = d(e, s, pool)?;
        let mut deriv = q;
        for i in 0..j {
            let low = crate::diff::undefined::build(inner, &[j - 1 - i], pool);
            let sign = if i % 2 == 0 { 1 } else { -1 };
            terms.push(pool.mul(vec![pool.integer(sign), deriv, low]));
            deriv = d(deriv, var, pool)?;
        }
        let sign = if j % 2 == 0 { 1 } else { -1 };
        residual.push(pool.mul(vec![pool.integer(sign), deriv]));
    }
    if !is_zero(pool.add(residual), pool) {
        return None;
    }
    Some(simplify(pool.add(terms), pool).value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::undefined::derivative_of;

    fn setup() -> (ExprPool, ExprId, ExprId) {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let f = pool.func("f", vec![t]);
        (pool, t, f)
    }

    fn int(e: ExprId, t: ExprId, pool: &ExprPool) -> Result<ExprId, IntegrationError> {
        integrate(e, t, pool).map(|r| r.value)
    }

    #[test]
    fn derivative_integrates_to_the_function() {
        let (pool, t, f) = setup();
        let f1 = derivative_of(f, &[1], &pool).unwrap();
        let f2 = derivative_of(f, &[2], &pool).unwrap();
        assert_eq!(int(f1, t, &pool).unwrap(), f);
        assert_eq!(int(f2, t, &pool).unwrap(), f1);
    }

    #[test]
    fn chain_rule_patterns() {
        let (pool, t, f) = setup();
        let f1 = derivative_of(f, &[1], &pool).unwrap();
        let sq = simplify(pool.mul(vec![f, f1]), &pool).value;
        let got = int(sq, t, &pool).unwrap();
        let want = simplify(
            pool.mul(vec![pool.rational(1, 2), pool.pow(f, pool.integer(2))]),
            &pool,
        )
        .value;
        assert_eq!(got, want);
        let logd = simplify(pool.mul(vec![f1, pool.pow(f, pool.integer(-1))]), &pool).value;
        assert_eq!(int(logd, t, &pool).unwrap(), pool.func("log", vec![f]));
    }

    #[test]
    fn polynomial_times_derivative_by_parts() {
        let (pool, t, f) = setup();
        let f1 = derivative_of(f, &[1], &pool).unwrap();
        let f2 = derivative_of(f, &[2], &pool).unwrap();
        // ∫ t f'' = t f' - f
        let got = int(simplify(pool.mul(vec![t, f2]), &pool).value, t, &pool).unwrap();
        let want = simplify(
            pool.add(vec![
                pool.mul(vec![t, f1]),
                pool.mul(vec![pool.integer(-1), f]),
            ]),
            &pool,
        )
        .value;
        assert_eq!(got, want);
        // ∫ (t f' + f) = t f, though neither term integrates alone
        let sum = pool.add(vec![pool.mul(vec![t, f1]), f]);
        assert_eq!(
            int(sum, t, &pool).unwrap(),
            simplify(pool.mul(vec![t, f]), &pool).value
        );
    }

    #[test]
    fn no_closed_form_is_a_refusal() {
        let (pool, t, f) = setup();
        let f1 = derivative_of(f, &[1], &pool).unwrap();
        assert!(int(f, t, &pool).is_err());
        assert!(int(pool.mul(vec![t, f]), t, &pool).is_err());
        assert!(int(pool.mul(vec![t, f1]), t, &pool).is_err());
        assert!(int(pool.func("sin", vec![f]), t, &pool).is_err());
    }

    #[test]
    fn definite_integral_is_endpoint_difference() {
        let (pool, t, f) = setup();
        let f1 = derivative_of(f, &[1], &pool).unwrap();
        let zero = pool.integer(0);
        let one = pool.integer(1);
        let r = super::super::engine::integrate_definite(f1, t, zero, one, &pool)
            .unwrap()
            .value;
        let want = simplify(
            pool.add(vec![
                pool.func("f", vec![one]),
                pool.mul(vec![pool.integer(-1), pool.func("f", vec![zero])]),
            ]),
            &pool,
        )
        .value;
        assert_eq!(r, want);
    }
}
