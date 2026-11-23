//! Numeric evaluation of the PDE layer's reserved heads, shared by the
//! `f64` evaluators (`eval_f64`, the JIT interpreter behind `eval_expr`).
//!
//! * `Sum(body, k, lo, hi)`: the bounds are evaluated first; when both are
//!   integers (within `1e-9`) and the range has at most
//!   [`super::sum::MAX_SUM_TERMS`] terms, each term `body[k := n]` is built by
//!   substitution and evaluated with a fresh memo (a node's value depends on
//!   `k`, so the caller's memo must not see it).  Otherwise no value.
//! * `Indexed(x, i)`: looked up in the environment like a symbol (bind
//!   `Indexed(x, 3)` to give `x₃` a value); after a `Sum` is expanded its
//!   indices are literals, so the lookup succeeds.
//! * `KroneckerDelta(i, j)`: `1.0` / `0.0` on integer arguments.
//! * `RootOfFamily(F, λ, n, branch)`: the certified enclosure's midpoint, with
//!   parameters bound from the environment.

use std::collections::HashMap;

use crate::kernel::{ExprId, ExprPool};

use super::root_family::ROOT_OF_FAMILY;
use super::sum::{sum_parts, INDEXED, KRONECKER_DELTA, MAX_SUM_TERMS, SUM};

/// Whether `Func(name, args)` is one of the heads evaluated here.
pub(crate) fn is_pde_head(name: &str, args: &[ExprId]) -> bool {
    matches!(
        (name, args.len()),
        (SUM, 4) | (INDEXED, 2) | (KRONECKER_DELTA, 2) | (ROOT_OF_FAMILY, 4)
    )
}

fn as_int(v: f64) -> Option<i64> {
    (v.is_finite() && (v - v.round()).abs() < 1e-9 && v.abs() < 9.0e15).then(|| v.round() as i64)
}

/// Evaluate one PDE head. `sub` evaluates an arbitrary expression in the
/// given environment with a fresh memo.
pub(crate) fn eval_head_f64(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    sub: &mut dyn FnMut(ExprId, &HashMap<ExprId, f64>) -> Option<f64>,
) -> Option<f64> {
    let (name, args) = pool.with(expr, |d| match d {
        crate::kernel::ExprData::Func { name, args } => Some((name.clone(), args.clone())),
        _ => None,
    })?;
    match name.as_str() {
        SUM => {
            let (body, k, lo, hi) = sum_parts(expr, pool)?;
            let lo = as_int(sub(lo, env)?)?;
            let hi = as_int(sub(hi, env)?)?;
            if hi < lo {
                return Some(0.0);
            }
            if (hi - lo) as u128 + 1 > MAX_SUM_TERMS as u128 {
                return None;
            }
            // The bound variable shadows any outer binding of `k`.
            let mut inner = env.clone();
            inner.remove(&k);
            let mut total = 0.0;
            for n in lo..=hi {
                let mut m = HashMap::new();
                m.insert(k, pool.integer(n));
                let term = crate::kernel::subs::subs(body, &m, pool);
                total += sub(term, &inner)?;
            }
            Some(total)
        }
        INDEXED => {
            if let Some(&v) = env.get(&expr) {
                return Some(v);
            }
            // An index that evaluates to an integer names a bound entry.
            let i = as_int(sub(args[1], env)?)?;
            let key = pool.func(INDEXED, vec![args[0], pool.integer(i)]);
            env.get(&key).copied()
        }
        KRONECKER_DELTA => {
            let a = as_int(sub(args[0], env)?)?;
            let b = as_int(sub(args[1], env)?)?;
            Some(if a == b { 1.0 } else { 0.0 })
        }
        ROOT_OF_FAMILY => super::root_family::root_family_f64(expr, env, pool),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    #[test]
    fn eval_expr_of_finite_and_symbolic_bound_sums() {
        let pool = ExprPool::new();
        let k = pool.symbol("k", Domain::Integer);
        let n = pool.symbol("n", Domain::Integer);
        let x = pool.symbol("x", Domain::Real);
        let body = pool.mul(vec![pool.pow(x, k), pool.pow(k, pool.integer(-1))]);
        let s = super::super::sum::sum(body, k, pool.integer(1), n, &pool).unwrap();
        let mut env = HashMap::new();
        env.insert(x, 0.5);
        env.insert(n, 30.0);
        let v = crate::jit::eval_interp(s, &env, &pool).unwrap();
        // Σ x^k/k → −log(1 − x)
        assert!((v - 2f64.ln()).abs() < 1e-9, "{v}");
        let v2 = crate::eval::eval_f64(s, &pool, &env).unwrap();
        assert!((v2 - v).abs() < 1e-12);
        // unbound n: no value
        env.remove(&n);
        assert!(crate::jit::eval_interp(s, &env, &pool).is_none());
    }

    #[test]
    fn indexed_lookup_and_delta() {
        let pool = ExprPool::new();
        let i = pool.symbol("i", Domain::Integer);
        let x = pool.symbol("x", Domain::Real);
        let d = pool.symbol("d", Domain::Positive);
        let xi = super::super::sum::indexed(x, i, &pool).unwrap();
        let s = super::super::sum::sum(pool.pow(xi, pool.integer(2)), i, pool.integer(1), d, &pool)
            .unwrap();
        let mut env = HashMap::new();
        env.insert(d, 3.0);
        for (j, v) in [(1, 1.0), (2, 2.0), (3, 3.0)] {
            env.insert(
                super::super::sum::indexed(x, pool.integer(j), &pool).unwrap(),
                v,
            );
        }
        let v = crate::jit::eval_interp(s, &env, &pool).unwrap();
        assert!((v - 14.0).abs() < 1e-12);
        let delta = pool.func(KRONECKER_DELTA, vec![i, pool.integer(2)]);
        env.insert(i, 2.0);
        assert_eq!(crate::jit::eval_interp(delta, &env, &pool).unwrap(), 1.0);
    }
}
