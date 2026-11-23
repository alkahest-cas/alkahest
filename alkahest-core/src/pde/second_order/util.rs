//! Small expression helpers shared by the second-order solvers.

use std::collections::HashMap;

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::super::error::PdeError;
use super::super::jet::symbols_of;

pub(crate) fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

/// Proved identically zero (exact reasoning only).
pub(crate) fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return true;
    }
    zero_test(s, pool, None).status == ZeroStatus::ProvedZero
}

/// Proved *not* identically zero.  Anything undecided counts as "maybe zero".
pub(crate) fn is_nonzero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return false;
    }
    if pool.with(s, |d| matches!(d, ExprData::Integer(_) | ExprData::Rational(_))) {
        return true;
    }
    zero_test(s, pool, None).status == ZeroStatus::ProvedNonzero
}

pub(crate) fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    simp(pool.mul(vec![a, pool.pow(b, pool.integer(-1))]), pool)
}

pub(crate) fn mul2(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, b])
}

pub(crate) fn exp(e: ExprId, pool: &ExprPool) -> ExprId {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return pool.integer(1);
    }
    pool.func("exp", vec![s])
}

/// `∂e/∂v`, simplified.
pub(crate) fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> Result<ExprId, PdeError> {
    crate::diff::diff(e, v, pool)
        .map(|r| simp(r.value, pool))
        .map_err(|err| PdeError::NotRationalInJets(err.to_string()))
}

/// An antiderivative `∫ e dv`, or `None` when the integrator declines.
pub(crate) fn integ(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return Some(pool.integer(0));
    }
    if free_of(s, v, pool) {
        return Some(simp(pool.mul(vec![s, v]), pool));
    }
    let r = crate::integrate::integrate(s, v, pool).ok()?;
    let r = simp(r.value, pool);
    // Belt and braces: the antiderivative must differentiate back.
    let back = d(r, v, pool).ok()?;
    if is_zero(sub(back, s, pool), pool) {
        Some(r)
    } else {
        None
    }
}

pub(crate) fn free_of(e: ExprId, v: ExprId, pool: &ExprPool) -> bool {
    !symbols_of(e, pool).contains(&v)
}

pub(crate) fn free_of_all(e: ExprId, vs: &[ExprId], pool: &ExprPool) -> bool {
    let syms = symbols_of(e, pool);
    vs.iter().all(|v| !syms.contains(v))
}

pub(crate) fn subs1(e: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    crate::kernel::subs::subs(e, &m, pool)
}

pub(crate) fn subs2(
    e: ExprId,
    a: (ExprId, ExprId),
    b: (ExprId, ExprId),
    pool: &ExprPool,
) -> ExprId {
    let mut m = HashMap::new();
    m.insert(a.0, a.1);
    m.insert(b.0, b.1);
    crate::kernel::subs::subs(e, &m, pool)
}

/// Every symbol name reachable from the expressions (for fresh names).
pub(crate) fn names_in(exprs: &[ExprId], pool: &ExprPool) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for &e in exprs {
        for s in symbols_of(e, pool) {
            pool.with(s, |d| {
                if let ExprData::Symbol { name, .. } = d {
                    out.insert(name.clone());
                }
            });
        }
        collect_func_names(e, pool, &mut out);
    }
    out
}

fn collect_func_names(e: ExprId, pool: &ExprPool, out: &mut std::collections::HashSet<String>) {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(name) = pool.with(n, |d| match d {
            ExprData::Func { name, .. } => Some(name.clone()),
            _ => None,
        }) {
            out.insert(name);
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
}

/// A fresh symbol named `base`, `base1`, `base2`, … avoiding `taken`.
pub(crate) fn fresh_symbol(
    base: &str,
    taken: &mut std::collections::HashSet<String>,
    pool: &ExprPool,
) -> ExprId {
    let name = fresh_name(base, taken);
    pool.symbol(name, Domain::Real)
}

pub(crate) fn fresh_name(base: &str, taken: &mut std::collections::HashSet<String>) -> String {
    let mut name = base.to_string();
    let mut k = 1;
    while taken.contains(&name) {
        name = format!("{base}{k}");
        k += 1;
    }
    taken.insert(name.clone());
    name
}

/// The multiplicative factors of `e`, with `exp(a + b)` split into
/// `exp(a)·exp(b)` and integer powers of products distributed — enough to
/// separate a coefficient into an `x`-only, a `y`-only and a mixed part.
pub(crate) fn factors(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    push_factors(e, pool, &mut out);
    out
}

fn push_factors(e: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(e) {
        ExprData::Mul(xs) => {
            for x in xs {
                push_factors(x, pool, out);
            }
        }
        ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
            let terms = pool.with(args[0], |d| match d {
                ExprData::Add(ts) => Some(ts.clone()),
                _ => None,
            });
            match terms {
                Some(ts) => {
                    for t in ts {
                        out.push(pool.func("exp", vec![t]));
                    }
                }
                None => out.push(e),
            }
        }
        ExprData::Pow { base, exp } => {
            let is_int = pool.with(exp, |d| matches!(d, ExprData::Integer(_)));
            let inner = pool.with(base, |d| match d {
                ExprData::Mul(xs) => Some(xs.clone()),
                _ => None,
            });
            match (is_int, inner) {
                (true, Some(xs)) => {
                    for x in xs {
                        let mut sub = Vec::new();
                        push_factors(x, pool, &mut sub);
                        for f in sub {
                            out.push(pool.pow(f, exp));
                        }
                    }
                }
                _ => {
                    // exp(a+b)^k
                    let is_exp = pool.with(base, |d| {
                        matches!(d, ExprData::Func { name, .. } if name == "exp")
                    });
                    if is_int && is_exp {
                        let mut sub = Vec::new();
                        push_factors(base, pool, &mut sub);
                        for f in sub {
                            out.push(pool.pow(f, exp));
                        }
                    } else {
                        out.push(e);
                    }
                }
            }
        }
        _ => out.push(e),
    }
}

/// Split `e` as `p(x) · q(y) · r(x, y)` by its factors.
pub(crate) fn split_xy(
    e: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> (ExprId, ExprId, ExprId) {
    let (mut p, mut q, mut r) = (Vec::new(), Vec::new(), Vec::new());
    for f in factors(e, pool) {
        let fx = free_of(f, x, pool);
        let fy = free_of(f, y, pool);
        if fy {
            p.push(f);
        } else if fx {
            q.push(f);
        } else {
            r.push(f);
        }
    }
    (
        simp(pool.mul(p), pool),
        simp(pool.mul(q), pool),
        simp(pool.mul(r), pool),
    )
}

/// The degree of `e` as a polynomial in `v` (coefficients free of `v`),
/// found by differentiating until zero, up to `max`.  `None` when `e` is
/// not a polynomial in `v` of degree ≤ `max`.
pub(crate) fn poly_degree_in(e: ExprId, v: ExprId, max: u32, pool: &ExprPool) -> Option<u32> {
    let mut cur = simp(e, pool);
    for k in 0..=max {
        if free_of(cur, v, pool) {
            return Some(k);
        }
        cur = d(cur, v, pool).ok()?;
        if is_zero(cur, pool) {
            return Some(k);
        }
    }
    None
}

/// `e` is `p + q·v` with `p`, `q` free of `v`: returns `(p, q)`.
pub(crate) fn linear_in(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let q = d(e, v, pool).ok()?;
    if !free_of(q, v, pool) {
        // the derivative may still be v-free after simplification
        let q2 = simp(q, pool);
        if !free_of(q2, v, pool) {
            return None;
        }
    }
    let p = simp(subs1(e, v, pool.integer(0), pool), pool);
    // confirm e == p + q v
    if !is_zero(sub(e, pool.add(vec![p, pool.mul(vec![q, v])]), pool), pool) {
        return None;
    }
    Some((p, q))
}

/// An exact square root of `e` when `e` is a perfect square: a rational
/// square, or a polynomial (integer coefficients after scaling) whose FLINT
/// factorisation has only even multiplicities.  A negative unit gives an
/// `I`.  `None` otherwise.
pub(crate) fn exact_sqrt(e: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let e = simp(e, pool);
    if let Some(r) = pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }) {
        let neg = r < 0;
        let a = r.clone().abs();
        let (num, den) = a.into_numer_denom();
        if num.is_perfect_square() && den.is_perfect_square() {
            let s = pool.rational(num.sqrt(), den.sqrt());
            return Some(if neg {
                simp(pool.mul(vec![pool.imaginary_unit(), s]), pool)
            } else {
                s
            });
        }
        return None;
    }
    // Polynomial in its symbols.
    let vars: Vec<ExprId> = symbols_of(e, pool).into_iter().collect();
    if vars.is_empty() || vars.iter().any(|&v| pool.is_imaginary_unit(v)) {
        return None;
    }
    // Clear rational denominators by scaling with a square.
    let mp = crate::poly::multipoly::MultiPoly::from_symbolic(e, vars.clone(), pool).ok()?;
    let f = crate::poly::factor::factor_multivariate_z(&mp).ok()?;
    let unit = f.unit.clone();
    let neg = unit < 0;
    let ua = unit.abs();
    if !ua.is_perfect_square() {
        return None;
    }
    let mut parts = vec![pool.integer(ua.sqrt())];
    for (p, m) in &f.factors {
        if m % 2 != 0 {
            return None;
        }
        parts.push(pool.pow(p.to_expr(pool), pool.integer(*m as i64 / 2)));
    }
    if neg {
        parts.push(pool.imaginary_unit());
    }
    Some(simp(pool.mul(parts), pool))
}

/// `√e`: exact when [`exact_sqrt`] finds it, the principal root otherwise.
pub(crate) fn sqrt(e: ExprId, pool: &ExprPool) -> ExprId {
    if let Some(s) = exact_sqrt(e, pool) {
        return s;
    }
    // −(perfect square) → I·root
    if let Some(s) = exact_sqrt(neg(e, pool), pool) {
        return simp(pool.mul(vec![pool.imaginary_unit(), s]), pool);
    }
    simp(pool.pow(e, pool.rational(1, 2)), pool)
}

/// Replace `I` by `−I` (complex conjugation when everything else is real).
pub(crate) fn conj(e: ExprId, pool: &ExprPool) -> ExprId {
    let i = pool.imaginary_unit();
    simp(subs1(e, i, neg(i, pool), pool), pool)
}

pub(crate) fn mentions_i(e: ExprId, pool: &ExprPool) -> bool {
    symbols_of(e, pool)
        .iter()
        .any(|&s| pool.is_imaginary_unit(s))
}
