//! Two square roots of linear radicands with the same slope, rationalised
//! together by `s = √(αx + β₁) + √(αx + β₂)`.
//!
//! With `c = β₁ − β₂`, `√(αx + β₁) − √(αx + β₂) = c/s`, so
//!
//! ```text
//! √(αx + β₁) = (s² + c)/(2s),   √(αx + β₂) = (s² − c)/(2s),
//! x = (((s² + c)/(2s))² − β₁)/α,
//! ```
//!
//! identities on the whole real domain of both roots (there `s > 0`).  Every
//! integrand `f(x, √(αx+β₁), √(αx+β₂))` — the roots may sit inside any
//! transcendental function, `asin(√x − √(x+1))` (Charlwood #3) — becomes a
//! function of `s` alone, and `∫f dx = ∫ f·(dx/ds) ds` is handed back to the
//! integrator.  The answer is substituted back and gated against the original
//! integrand; a miss is a decline and nothing more.

use std::collections::HashMap;

use rug::Rational;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::risch::poly_rde::{degree, expr_to_qpoly, rational_to_expr};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{integrate, is_free_of, verify_antiderivative_status};

pub(crate) const RADICAL_PAIR_RULE: &str = "int_radical_pair_substitution";

/// `(radicand, power k)` when `e` is `L^{k/2}` (`k` odd) or `sqrt(L)`, `L`
/// linear in `var` with rational coefficients `(β, α)`.
fn half_power(
    e: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, i64, Rational, Rational)> {
    let (base, k) = match pool.get(e) {
        ExprData::Func { name, args } if name == "sqrt" && args.len() == 1 => (args[0], 1),
        ExprData::Pow { base, exp } => match pool.get(exp) {
            ExprData::Rational(r) if *r.0.denom() == 2 => (base, r.0.numer().to_i64()?),
            _ => return None,
        },
        _ => return None,
    };
    let p = expr_to_qpoly(base, var, pool)?;
    if degree(&p) != 1 {
        return None;
    }
    Some((base, k, p[0].clone(), p[1].clone()))
}

fn collect(e: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<(Rational, Rational)>) {
    if is_free_of(e, var, pool) {
        return;
    }
    if let Some((_, _, beta, alpha)) = half_power(e, var, pool) {
        if !out.contains(&(beta.clone(), alpha.clone())) {
            out.push((beta, alpha));
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

/// Rewrite `e` with each `L_i^{k/2}` replaced by `root_i^k` and the bare
/// variable by `x_of_s`.  `None` on a node this rewrite does not understand.
fn rewrite(
    e: ExprId,
    var: ExprId,
    roots: &[(Rational, Rational, ExprId)],
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
    } else if let Some((_, k, beta, alpha)) = half_power(e, var, pool) {
        let (_, _, root) = roots.iter().find(|(b, a, _)| *b == beta && *a == alpha)?;
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
    let alpha = found[0].1.clone();
    let (b1, b2) = (found[0].0.clone(), found[1].0.clone());
    let c = Rational::from(&b1 - &b2);
    let s = pool.symbol("__radpair_s", crate::kernel::Domain::Real);
    let two_s_inv = pool.pow(pool.mul(vec![pool.integer(2_i32), s]), pool.integer(-1_i32));
    let s2 = pool.pow(s, pool.integer(2_i32));
    let c_e = rational_to_expr(&c, pool);
    let root1 = pool.mul(vec![pool.add(vec![s2, c_e]), two_s_inv]);
    let root2 = pool.mul(vec![
        pool.add(vec![s2, pool.mul(vec![pool.integer(-1_i32), c_e])]),
        two_s_inv,
    ]);
    let x_of_s = simplify(
        pool.mul(vec![
            pool.add(vec![
                pool.pow(root1, pool.integer(2_i32)),
                rational_to_expr(&Rational::from(-&b1), pool),
            ]),
            rational_to_expr(&(Rational::from(1) / alpha.clone()), pool),
        ]),
        pool,
    )
    .value;
    let roots = [
        (b1.clone(), alpha.clone(), root1),
        (b2.clone(), alpha.clone(), root2),
    ];
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
    // s = √(αx + β₁) + √(αx + β₂)
    let lin = |b: &Rational| {
        pool.add(vec![
            pool.mul(vec![rational_to_expr(&alpha, pool), var]),
            rational_to_expr(b, pool),
        ])
    };
    let s_of_x = pool.add(vec![
        pool.pow(lin(&b1), pool.rational(1, 2)),
        pool.pow(lin(&b2), pool.rational(1, 2)),
    ]);
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

    /// One radical, or two with different slopes: not this route.
    #[test]
    fn declines_without_a_matching_pair() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        for src in ["asin(sqrt(x))", "sqrt(x)*sqrt(2*x + 1)*log(x)"] {
            let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
            assert!(try_radical_pair(f, x, &pool).is_none(), "{src}");
        }
    }
}
