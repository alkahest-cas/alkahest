//! Two square roots whose radicands differ by a constant, rationalised
//! together by `s = √R₁ + √R₂`.
//!
//! The radicands are linear-fractional, `Rᵢ = (aᵢx + bᵢ)/(cx + d)` with one
//! denominator, and `R₁ − R₂ = c₀` is a constant.  Then `√R₁ − √R₂ = c₀/s`, so
//!
//! ```text
//! √R₁ = (s² + c₀)/(2s),   √R₂ = (s² − c₀)/(2s),
//! R₁(x) = ((s² + c₀)/(2s))²   ⇒   x rational in s,
//! ```
//!
//! identities on the whole real domain of both roots (there `s > 0`).
//! `√x, √(x+1)` and `√(1/s + 1), √(1/s − 1)` are the two shapes met so far.  Every
//! integrand `f(x, √(αx+β₁), √(αx+β₂))` — the roots may sit inside any
//! transcendental function, `asin(√x − √(x+1))` (Charlwood #3) — becomes a
//! function of `s` alone, and `∫f dx = ∫ f·(dx/ds) ds` is handed back to the
//! integrator.  The answer is substituted back and gated against the original
//! integrand; a miss is a decline and nothing more.

use std::collections::HashMap;

use rug::Rational;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::risch::poly_rde::{
    degree, poly_scale, qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use crate::integrate::risch::rational_rde::{expr_to_qrational, poly_sub};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{integrate, is_free_of, verify_antiderivative_status};

pub(crate) const RADICAL_PAIR_RULE: &str = "int_radical_pair_substitution";

/// A linear-fractional radicand `n/d` (`deg ≤ 1` each, `d` monic).
type Radicand = (QPoly, QPoly);

/// `(radicand, k)` when `e` is `R^{k/2}` (`k` odd) or `sqrt(R)` with `R`
/// linear-fractional in `var` with rational coefficients.
fn half_power(e: ExprId, var: ExprId, pool: &ExprPool) -> Option<(Radicand, i64)> {
    let (base, k) = match pool.get(e) {
        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => (args[0], 1),
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Rational(r) if *r.0.denom() == 2 => (base, r.0.numer().to_i64()?),
            _ => return None,
        },
        _ => return None,
    };
    let (n, d) = expr_to_qrational(base, var, pool)?;
    let (n, d) = (trim(n), trim(d));
    if degree(&n) > 1 || degree(&d) > 1 || degree(&n).max(degree(&d)) < 1 {
        return None;
    }
    let lc = d[d.len() - 1].clone();
    let inv = Rational::from(1) / lc;
    Some(((trim(poly_scale(&n, &inv)), trim(poly_scale(&d, &inv))), k))
}

fn collect(e: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<Radicand>) {
    if is_free_of(e, var, pool) {
        return;
    }
    if let Some((r, _)) = half_power(e, var, pool) {
        if !out.contains(&r) {
            out.push(r);
        }
        return;
    }
    match pool.get(e) {
        ExprData::Func { args, .. } | ExprData::Add(args) | ExprData::Mul(args) => {
            for a in args {
                collect(a, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            collect(base, var, pool, out);
            collect(exp, var, pool, out);
        }
        _ => {}
    }
}

/// Rewrite `e` with each `Rᵢ^{k/2}` replaced by `rootᵢ^k` and the bare
/// variable by `x_of_s`.  `None` on a node this rewrite does not understand.
fn rewrite(
    e: ExprId,
    var: ExprId,
    roots: &[(Radicand, ExprId)],
    x_of_s: ExprId,
    pool: &ExprPool,
    memo: &mut HashMap<ExprId, ExprId>,
) -> Option<ExprId> {
    if is_free_of(e, var, pool) {
        return Some(e);
    }
    if let Some(&r) = memo.get(&e) {
        return Some(r);
    }
    let out = if e == var {
        x_of_s
    } else if let Some((rad, k)) = half_power(e, var, pool) {
        let (_, root) = roots.iter().find(|(r, _)| *r == rad)?;
        pool.pow(*root, pool.integer(k))
    } else {
        match pool.get(e) {
            ExprData::Func { name, args } => {
                let a = args
                    .iter()
                    .map(|&a| rewrite(a, var, roots, x_of_s, pool, memo))
                    .collect::<Option<Vec<_>>>()?;
                pool.func(name, a)
            }
            ExprData::Add(args) => pool.add(
                args.iter()
                    .map(|&a| rewrite(a, var, roots, x_of_s, pool, memo))
                    .collect::<Option<Vec<_>>>()?,
            ),
            ExprData::Mul(args) => pool.mul(
                args.iter()
                    .map(|&a| rewrite(a, var, roots, x_of_s, pool, memo))
                    .collect::<Option<Vec<_>>>()?,
            ),
            ExprData::Pow { base, exp } => pool.pow(
                rewrite(base, var, roots, x_of_s, pool, memo)?,
                rewrite(exp, var, roots, x_of_s, pool, memo)?,
            ),
            _ => return None,
        }
    };
    memo.insert(e, out);
    Some(out)
}

/// Put every function argument that is rational in `s` over a common
/// denominator, so `asin((s²−1)/(2s) − (s²+1)/(2s))` reads `asin(−1/s)`.
fn cancel_args(e: ExprId, s: ExprId, pool: &ExprPool) -> ExprId {
    if is_free_of(e, s, pool) {
        return e;
    }
    match pool.get(e) {
        ExprData::Func { name, args } => {
            let a = args
                .iter()
                .map(|&a| {
                    let a = cancel_args(a, s, pool);
                    crate::poly::cancel::cancel(a, vec![s], pool)
                        .map(|c| simplify(c, pool).value)
                        .unwrap_or(a)
                })
                .collect();
            pool.func(name, a)
        }
        ExprData::Add(args) => pool.add(args.iter().map(|&a| cancel_args(a, s, pool)).collect()),
        ExprData::Mul(args) => pool.mul(args.iter().map(|&a| cancel_args(a, s, pool)).collect()),
        ExprData::Pow { base, exp } => {
            pool.pow(cancel_args(base, s, pool), cancel_args(exp, s, pool))
        }
        _ => e,
    }
}

pub(crate) fn try_radical_pair(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    let mut found = Vec::new();
    collect(expr, var, pool, &mut found);
    if found.len() != 2 || found[0].1 != found[1].1 {
        return None;
    }
    let ((n1, d), (n2, _)) = (found[0].clone(), found[1].clone());
    // R₁ − R₂ = (n₁ − n₂)/d must be a constant c₀.
    let diff = trim(poly_sub(&n1, &n2));
    let c = if degree(&d) == 0 && degree(&diff) <= 0 {
        diff.first().cloned().unwrap_or_else(|| Rational::from(0))
    } else if degree(&diff) == degree(&d) {
        let c = diff[diff.len() - 1].clone() / d[d.len() - 1].clone();
        if !trim(poly_sub(&diff, &poly_scale(&d, &c))).is_empty() {
            return None;
        }
        c
    } else {
        return None;
    };
    if c == 0 {
        return None;
    }
    let s = pool.symbol("__radpair_s", crate::kernel::Domain::Real);
    let two_s_inv = pool.pow(pool.mul(vec![pool.integer(2_i32), s]), pool.integer(-1_i32));
    let s2 = pool.pow(s, pool.integer(2_i32));
    let c_e = rational_to_expr(&c, pool);
    let root1 = pool.mul(vec![pool.add(vec![s2, c_e]), two_s_inv]);
    let root2 = pool.mul(vec![
        pool.add(vec![s2, pool.mul(vec![pool.integer(-1_i32), c_e])]),
        two_s_inv,
    ]);
    // n₁(x) = T·d(x), T = root₁²:  x = (T·d₀ − n₁₀)/(n₁₁ − T·d₁).
    let coeff =
        |p: &QPoly, i: usize| rational_to_expr(&p.get(i).cloned().unwrap_or_default(), pool);
    let t = pool.pow(root1, pool.integer(2_i32));
    let neg = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
    let x_of_s = simplify(
        pool.mul(vec![
            pool.add(vec![pool.mul(vec![t, coeff(&d, 0)]), neg(coeff(&n1, 0))]),
            pool.pow(
                pool.add(vec![coeff(&n1, 1), neg(pool.mul(vec![t, coeff(&d, 1)]))]),
                pool.integer(-1_i32),
            ),
        ]),
        pool,
    )
    .value;
    let roots = [(found[0].clone(), root1), (found[1].clone(), root2)];
    let mut memo = HashMap::new();
    let f_s = rewrite(expr, var, &roots, x_of_s, pool, &mut memo)?;
    let dx_ds = crate::diff::diff(x_of_s, s, pool).ok()?.value;
    let g = simplify(pool.mul(vec![f_s, dx_ds]), pool).value;
    let g = cancel_args(g, s, pool);
    if !is_free_of(g, var, pool) {
        return None;
    }
    crate::budget::check().ok()?;
    let inner = integrate(g, s, pool).ok()?;
    // s = √R₁ + √R₂
    let rad = |r: &Radicand| {
        pool.pow(
            pool.mul(vec![
                qpoly_to_expr(&r.0, var, pool),
                pool.pow(qpoly_to_expr(&r.1, var, pool), pool.integer(-1_i32)),
            ]),
            pool.rational(1, 2),
        )
    };
    let s_of_x = pool.add(vec![rad(&found[0]), rad(&found[1])]);
    let back = HashMap::from([(s, s_of_x)]);
    let f = simplify(crate::kernel::subs(inner.value, &back, pool), pool).value;
    if !is_free_of(f, s, pool) || verify_antiderivative_status(f, expr, var, pool).is_none() {
        return None;
    }
    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple(RADICAL_PAIR_RULE, expr, f));
    Some(DerivedExpr::with_log(f, log.merge(inner.log)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Charlwood #3: `∫ −asin(√x − √(x+1)) dx`.
    #[test]
    fn charlwood_3() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let f = crate::parse::parse("-asin(sqrt(x) - sqrt(x + 1))", &pool, &mut syms).unwrap();
        let r = try_radical_pair(f, x, &pool).expect("closes");
        assert!(verify_antiderivative_status(r.value, f, x, &pool).is_some());
    }

    /// The Möbius shape: `√(1/s + 1)`, `√(1/s − 1)` (Charlwood #45 after
    /// `s = cos x`, without its `1/√(1 − s²)`).
    #[test]
    fn reciprocal_radicands() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let f = crate::parse::parse("sqrt(1/x + 1)*sqrt(1/x - 1)/x", &pool, &mut syms).unwrap();
        let r = try_radical_pair(f, x, &pool).expect("closes");
        assert!(verify_antiderivative_status(r.value, f, x, &pool).is_some());
    }

    /// One radical, or two whose radicands differ by a non-constant: not
    /// this route.
    #[test]
    fn declines_without_a_matching_pair() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        for src in [
            "asin(sqrt(x))",
            "sqrt(x)*sqrt(2*x + 1)*log(x)",
            "sqrt(x)*sqrt(1/x + 1)",
        ] {
            let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
            assert!(try_radical_pair(f, x, &pool).is_none(), "{src}");
        }
    }
}
