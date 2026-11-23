//! The reserved head `Integral`: unevaluated integrals as first-class
//! values.
//!
//! The head has two forms:
//!
//! * `Integral(f, t, lo, hi)` is `∫_{lo}^{hi} f dt`. It binds `t` in `f`, like
//!   [`crate::pde::sum`]'s `Sum`, so `subs` shadows and renames. The
//!   bounds are free.
//! * `Integral(f, x)` is the indefinite integral, *defined* as
//!   `Integral(f[x := t], t, 0, x)`.  `x` is free. Substituting a non-symbol for
//!   `x` produces the definite form with that upper bound.
//!
//! Calculus rules, all exact:
//!
//! * Leibniz: `d/dv ∫_{lo}^{hi} f dt = f(hi)·hi′ − f(lo)·lo′ + ∫_{lo}^{hi} ∂f/∂v dt`.
//! * The fundamental theorem for the indefinite form: `d/dx Integral(f, x) =
//!   f`, and `d/dv Integral(f, x) = Integral(∂f/∂v, x)` for `v ≠ x`.
//!
//! So verification by substitution stays symbolic: an answer containing
//! `Integral(…)` is checked through these rules, with no quadrature.
//! [`combine_integrals`] merges `Σ cᵢ·∫ fᵢ dt` with `t`-free coefficients into
//! one integral and drops integrals whose integrand is proved zero. Without
//! that, two integrals that cancel would stay independent atoms for the
//! zero test.
//!
//! Numeric evaluation (f64 evaluators) uses composite Gauss–Legendre
//! quadrature. It serves sampling only, and a numeric value never certifies.

use std::collections::{BTreeMap, HashMap};

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::error::PdeError;
use super::jet::symbols_of;

/// The head name.
pub const INTEGRAL: &str = "Integral";

fn is_symbol(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| matches!(d, ExprData::Symbol { .. }))
}

fn mentions(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    symbols_of(e, pool).contains(&v)
}

/// `∫_{lo}^{hi} body dt`.
///
/// # Errors
///
/// `E-PDE-054` when `t` is not a symbol or a bound mentions `t`.
pub fn integral(
    body: ExprId,
    t: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    if !is_symbol(t, pool) {
        return Err(PdeError::MalformedIntegral(format!(
            "the integration variable {} is not a symbol",
            pool.display(t)
        )));
    }
    if mentions(lo, t, pool) || mentions(hi, t, pool) {
        return Err(PdeError::MalformedIntegral(format!(
            "a bound of the integral over {} mentions {}",
            pool.display(t),
            pool.display(t)
        )));
    }
    Ok(pool.func(INTEGRAL, vec![body, t, lo, hi]))
}

/// The indefinite `Integral(body, x)` (`= ∫_0^x body[x := t] dt`).
///
/// # Errors
///
/// `E-PDE-054` when `x` is not a symbol.
pub fn indefinite_integral(body: ExprId, x: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    if !is_symbol(x, pool) {
        return Err(PdeError::MalformedIntegral(format!(
            "the integration variable {} is not a symbol",
            pool.display(x)
        )));
    }
    Ok(pool.func(INTEGRAL, vec![body, x]))
}

/// `(body, t, lo, hi)` of a definite `Integral`.
pub fn integral_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    let args = pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == INTEGRAL && args.len() == 4 => Some(args.clone()),
        _ => None,
    })?;
    is_symbol(args[1], pool).then_some((args[0], args[1], args[2], args[3]))
}

/// `(body, x)` of an indefinite `Integral`.
pub fn indefinite_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let args = pool.with(e, |d| match d {
        ExprData::Func { name, args } if name == INTEGRAL && args.len() == 2 => Some(args.clone()),
        _ => None,
    })?;
    is_symbol(args[1], pool).then_some((args[0], args[1]))
}

/// A symbol named like `stem` that occurs in none of `avoid`.
pub(crate) fn fresh_dummy(stem: &str, avoid: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut taken = std::collections::HashSet::new();
    for &a in avoid {
        for s in symbols_of(a, pool) {
            if let ExprData::Symbol { name, .. } = pool.get(s) {
                taken.insert(name);
            }
        }
    }
    let mut k = 0usize;
    loop {
        let name = if k == 0 {
            stem.to_string()
        } else {
            format!("{stem}{k}")
        };
        if !taken.contains(&name) {
            return pool.symbol(name, Domain::Real);
        }
        k += 1;
    }
}

/// The indefinite form as its definite equivalent
/// `Integral(body[x := t], t, 0, x)`.
pub fn to_definite(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (body, x) = indefinite_parts(e, pool)?;
    let t = fresh_dummy("_t", &[body, x], pool);
    let mut m = HashMap::new();
    m.insert(x, t);
    let b = crate::kernel::subs::subs(body, &m, pool);
    Some(pool.func(INTEGRAL, vec![b, t, pool.integer(0), x]))
}

/// `d/dv` of an `Integral` node (either form), through `diff`'s callback
/// `d` for sub-expressions.  `None` when `e` is not an `Integral`.
pub(crate) fn derivative<E, F>(
    e: ExprId,
    v: ExprId,
    pool: &ExprPool,
    mut d: F,
) -> Option<Result<ExprId, E>>
where
    F: FnMut(ExprId) -> Result<ExprId, E>,
{
    let zero = pool.integer(0);
    if let Some((body, x)) = indefinite_parts(e, pool) {
        if v == x {
            return Some(Ok(body));
        }
        if !mentions(body, v, pool) {
            return Some(Ok(zero));
        }
        return Some(d(body).map(|db| {
            if db == zero {
                zero
            } else {
                pool.func(INTEGRAL, vec![db, x])
            }
        }));
    }
    let (body, t, lo, hi) = integral_parts(e, pool)?;
    let mut terms = Vec::new();
    let at = |b: ExprId, val: ExprId| {
        let mut m = HashMap::new();
        m.insert(t, val);
        crate::kernel::subs::subs(b, &m, pool)
    };
    let r = (|| -> Result<ExprId, E> {
        if mentions(hi, v, pool) {
            let dh = d(hi)?;
            terms.push(pool.mul(vec![at(body, hi), dh]));
        }
        if mentions(lo, v, pool) {
            let dl = d(lo)?;
            terms.push(pool.mul(vec![pool.integer(-1), at(body, lo), dl]));
        }
        if t != v && mentions(body, v, pool) {
            // Rename t away from v so it cannot capture it.
            let (b2, t2) = if mentions(v, t, pool) {
                let nt = fresh_dummy("_t", &[body, v], pool);
                let mut m = HashMap::new();
                m.insert(t, nt);
                (crate::kernel::subs::subs(body, &m, pool), nt)
            } else {
                (body, t)
            };
            let db = d(b2)?;
            if db != zero {
                terms.push(pool.func(INTEGRAL, vec![db, t2, lo, hi]));
            }
        }
        Ok(if terms.is_empty() {
            zero
        } else {
            pool.add(terms)
        })
    })();
    Some(r)
}

/// Rename every definite integral's bound variable canonically
/// (`__alkahest_t{depth}`), so alpha-equivalent integrals are one atom.
pub fn alpha_normalize_integrals(expr: ExprId, pool: &ExprPool) -> ExprId {
    fn rec(
        e: ExprId,
        depth: usize,
        pool: &ExprPool,
        memo: &mut HashMap<(ExprId, usize), ExprId>,
    ) -> ExprId {
        if let Some(&r) = memo.get(&(e, depth)) {
            return r;
        }
        let out = if let Some((body, t, lo, hi)) = integral_parts(e, pool) {
            let lo = rec(lo, depth, pool, memo);
            let hi = rec(hi, depth, pool, memo);
            let canon = pool.symbol(format!("__alkahest_t{depth}"), Domain::Real);
            let body = if canon == t {
                body
            } else {
                let mut m = HashMap::new();
                m.insert(t, canon);
                crate::kernel::subs::subs(body, &m, pool)
            };
            let body = rec(body, depth + 1, pool, memo);
            pool.func(INTEGRAL, vec![body, canon, lo, hi])
        } else {
            match pool.get(e) {
                ExprData::Add(xs) => {
                    pool.add(xs.iter().map(|&x| rec(x, depth, pool, memo)).collect())
                }
                ExprData::Mul(xs) => {
                    pool.mul(xs.iter().map(|&x| rec(x, depth, pool, memo)).collect())
                }
                ExprData::Pow { base, exp } => {
                    let b = rec(base, depth, pool, memo);
                    let x = rec(exp, depth, pool, memo);
                    pool.pow(b, x)
                }
                ExprData::Func { name, args } => {
                    let a: Vec<ExprId> = args.iter().map(|&x| rec(x, depth, pool, memo)).collect();
                    pool.func(name, a)
                }
                _ => e,
            }
        };
        memo.insert((e, depth), out);
        out
    }
    rec(expr, 0, pool, &mut HashMap::new())
}

/// `Σ cᵢ·Integral(fᵢ, t, lo, hi) → Integral(Σ cᵢ fᵢ, t, lo, hi)` for
/// coefficients free of `t` (and likewise for indefinite integrals in the
/// same variable); integrals whose integrand is proved zero are dropped.
/// Apply after [`alpha_normalize_integrals`].
pub fn combine_integrals(expr: ExprId, pool: &ExprPool) -> ExprId {
    let has = pool.with(expr, |d| match d {
        ExprData::Func { name, .. } => name == INTEGRAL,
        _ => false,
    });
    let rebuilt = match pool.get(expr) {
        ExprData::Add(xs) => {
            let xs: Vec<ExprId> = xs.iter().map(|&x| combine_integrals(x, pool)).collect();
            // key → accumulated integrand terms
            let mut groups: BTreeMap<Vec<ExprId>, Vec<ExprId>> = BTreeMap::new();
            let mut rest = Vec::new();
            for x in xs {
                match split_term(x, pool) {
                    Some((key, contrib)) => groups.entry(key).or_default().push(contrib),
                    None => rest.push(x),
                }
            }
            for (key, parts) in groups {
                let body = simplify(pool.add(parts), pool).value;
                let body = super::solve::tidy(body, pool);
                if is_zero(body, pool) {
                    continue;
                }
                let mut args = vec![body];
                args.extend(key);
                rest.push(pool.func(INTEGRAL, args));
            }
            if rest.is_empty() {
                pool.integer(0)
            } else {
                pool.add(rest)
            }
        }
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| combine_integrals(x, pool)).collect()),
        ExprData::Pow { base, exp } => {
            pool.pow(combine_integrals(base, pool), combine_integrals(exp, pool))
        }
        ExprData::Func { name, args } if name != INTEGRAL => pool.func(
            name,
            args.iter().map(|&a| combine_integrals(a, pool)).collect(),
        ),
        _ => expr,
    };
    if has {
        // A lone integral with a zero integrand.
        if let Some((key, contrib)) = split_term(expr, pool) {
            let _ = key;
            if is_zero(contrib, pool) {
                return pool.integer(0);
            }
        }
    }
    rebuilt
}

fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simplify(e, pool).value;
    s == pool.integer(0)
        || crate::simplify::zero::zero_test(s, pool, None).status
            == crate::simplify::zero::ZeroStatus::ProvedZero
}

/// `c·Integral(f, t, lo, hi)` → (`[t, lo, hi]`, `c·f`) when `c` is free of
/// `t`; for the indefinite form the key is `[x]` and `c` must be free of `x`.
fn split_term(x: ExprId, pool: &ExprPool) -> Option<(Vec<ExprId>, ExprId)> {
    let factors = match pool.get(x) {
        ExprData::Mul(fs) => fs,
        _ => vec![x],
    };
    let pos = factors.iter().position(|&f| {
        pool.with(
            f,
            |d| matches!(d, ExprData::Func { name, .. } if name == INTEGRAL),
        )
    })?;
    let node = factors[pos];
    let coef: Vec<ExprId> = factors
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != pos)
        .map(|(_, &f)| f)
        .collect();
    let c = pool.mul(coef);
    if let Some((body, t, lo, hi)) = integral_parts(node, pool) {
        if mentions(c, t, pool) {
            return None;
        }
        return Some((vec![t, lo, hi], pool.mul(vec![c, body])));
    }
    if let Some((body, xv)) = indefinite_parts(node, pool) {
        if mentions(c, xv, pool) {
            return None;
        }
        return Some((vec![xv], pool.mul(vec![c, body])));
    }
    None
}

/// Whether `e` contains an `Integral` node.
pub fn has_integral(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let hit = pool.with(
            n,
            |d| matches!(d, ExprData::Func { name, .. } if name == INTEGRAL),
        );
        if hit {
            return true;
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    false
}

/// f64 value of an `Integral` node by composite Gauss–Legendre quadrature;
/// `sub` evaluates sub-expressions in an environment.
pub(crate) fn integral_f64(
    expr: ExprId,
    env: &HashMap<ExprId, f64>,
    pool: &ExprPool,
    sub: &mut super::numeric::SubEval<'_>,
) -> Option<f64> {
    let (body, t, lo, hi) = match integral_parts(expr, pool) {
        Some(p) => p,
        None => {
            let (body, x) = indefinite_parts(expr, pool)?;
            let hi = *env.get(&x)?;
            return quad(body, x, 0.0, hi, env, sub);
        }
    };
    let a = sub(lo, env)?;
    let b = sub(hi, env)?;
    quad(body, t, a, b, env, sub)
}

fn quad(
    body: ExprId,
    t: ExprId,
    a: f64,
    b: f64,
    env: &HashMap<ExprId, f64>,
    sub: &mut super::numeric::SubEval<'_>,
) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    // 5-point Gauss–Legendre on 32 panels.
    const X: [f64; 5] = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683,
        0.0,
        0.538_469_310_105_683,
        0.906_179_845_938_664,
    ];
    const W: [f64; 5] = [
        0.236_926_885_056_189,
        0.478_628_670_499_366,
        0.568_888_888_888_889,
        0.478_628_670_499_366,
        0.236_926_885_056_189,
    ];
    let panels = 32;
    let h = (b - a) / panels as f64;
    let mut inner = env.clone();
    let mut total = 0.0;
    for p in 0..panels {
        let mid = a + (p as f64 + 0.5) * h;
        for (x, w) in X.iter().zip(W.iter()) {
            inner.insert(t, mid + 0.5 * h * x);
            total += w * sub(body, &inner)?;
        }
    }
    let v = total * 0.5 * h;
    v.is_finite().then_some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftc_leibniz_and_numeric() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        // ∫_1^x exp(t² y) dt
        let body = pool.func("exp", vec![pool.mul(vec![pool.pow(t, pool.integer(2)), y])]);
        let i = integral(body, t, pool.integer(1), x, &pool).unwrap();
        let dx = crate::diff::diff(i, x, &pool).unwrap().value;
        let want = pool.func("exp", vec![pool.mul(vec![pool.pow(x, pool.integer(2)), y])]);
        assert_eq!(simplify(dx, &pool).value, simplify(want, &pool).value);
        let dy = crate::diff::diff(i, y, &pool).unwrap().value;
        assert!(has_integral(dy, &pool));
        let mut env = HashMap::new();
        env.insert(x, 2.0);
        env.insert(y, 0.0);
        let v = crate::jit::eval_interp(i, &env, &pool).unwrap();
        assert!((v - 1.0).abs() < 1e-10);
        // Indefinite: d/dx Integral(f(x), x) = f(x), with f undefined.
        let fx = pool.func("f", vec![x]);
        let ii = indefinite_integral(fx, x, &pool).unwrap();
        assert_eq!(crate::diff::diff(ii, x, &pool).unwrap().value, fx);
        // Cancellation through combine.
        let a = alpha_normalize_integrals(
            pool.add(vec![i, pool.mul(vec![pool.integer(-1), i])]),
            &pool,
        );
        assert_eq!(
            combine_integrals(simplify(a, &pool).value, &pool),
            pool.integer(0)
        );
        // subs into the free bound; the bound variable is shadowed.
        let mut m = HashMap::new();
        m.insert(t, pool.integer(5));
        assert_eq!(crate::kernel::subs::subs(i, &m, &pool), i);
    }
}
