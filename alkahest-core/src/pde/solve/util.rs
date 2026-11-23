//! Small algebra helpers shared by the `pdsolve` tiers: zero tests, freeness,
//! isolation of one symbol from an equation, quadrature, and numeric probes
//! used **only** to choose between candidates (never to certify anything).

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::diff::diff_impl::diff;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::super::jet::{push_children, symbols_of};

pub(crate) fn simp(e: ExprId, pool: &ExprPool) -> ExprId {
    simplify(e, pool).value
}

pub(crate) fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}

pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}

pub(crate) fn recip(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.pow(e, pool.integer(-1))
}

pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, recip(b, pool)])
}

/// `simplify(∂e/∂v)`; `None` when `diff` refuses.
pub(crate) fn d(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    diff(e, v, pool).ok().map(|r| simp(r.value, pool))
}

/// Proved identically zero (symbolic; numeric evidence never counts).
pub(crate) fn is_zero(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return true;
    }
    zero_test(s, pool, None).status == ZeroStatus::ProvedZero
}

pub(crate) fn is_number(e: ExprId, pool: &ExprPool) -> bool {
    pool.with(e, |d| {
        matches!(
            d,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_)
        )
    })
}

pub(crate) fn symbol_name(e: ExprId, pool: &ExprPool) -> Option<String> {
    pool.with(e, |d| match d {
        ExprData::Symbol { name, .. } => Some(name.clone()),
        _ => None,
    })
}

/// Whether `target` occurs as a sub-expression of `e` (structural).
pub(crate) fn contains(e: ExprId, target: ExprId, pool: &ExprPool) -> bool {
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if n == target {
            return true;
        }
        if !seen.insert(n) {
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

pub(crate) fn free_of(e: ExprId, vars: &[ExprId], pool: &ExprPool) -> bool {
    vars.iter().all(|&v| !contains(e, v, pool))
}

pub(crate) fn symbols(e: ExprId, pool: &ExprPool) -> BTreeSet<ExprId> {
    symbols_of(e, pool)
}

/// Whether `e` contains an applied undefined function (arbitrary function or
/// unknown).
pub(crate) fn has_undefined(e: ExprId, pool: &ExprPool) -> bool {
    crate::diff::undefined::has_undefined_function(e, pool)
}

/// Every name used (symbols and functions) in `exprs`.
pub(crate) fn names_in(exprs: &[ExprId], pool: &ExprPool) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack: Vec<ExprId> = exprs.to_vec();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        pool.with(n, |d| match d {
            ExprData::Symbol { name, .. } | ExprData::Func { name, .. } => {
                out.insert(name.clone());
            }
            _ => {}
        });
        push_children(n, pool, &mut stack);
    }
    out
}

/// A fresh real symbol `{stem}{k}` whose name is not in `taken` (which is
/// updated).
pub(crate) fn fresh_symbol(stem: &str, taken: &mut HashSet<String>, pool: &ExprPool) -> ExprId {
    let mut k = 1usize;
    loop {
        let name = format!("{stem}{k}");
        if taken.insert(name.clone()) {
            return pool.symbol(name, Domain::Real);
        }
        k += 1;
    }
}

pub(crate) fn subs1(e: ExprId, from: ExprId, to: ExprId, pool: &ExprPool) -> ExprId {
    let mut m = HashMap::new();
    m.insert(from, to);
    crate::kernel::subs::subs(e, &m, pool)
}

pub(crate) fn subs_map(e: ExprId, m: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    crate::kernel::subs::subs(e, m, pool)
}

/// Integer value of a numeric node.
pub(crate) fn as_rational(e: ExprId, pool: &ExprPool) -> Option<rug::Rational> {
    pool.with(e, |d| match d {
        ExprData::Integer(n) => Some(rug::Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// Isolation: solve `eq = 0` for one symbol
// ---------------------------------------------------------------------------

/// Candidate solutions of `eq = 0` for the symbol `v`, by (1) linearity,
/// (2) structural peeling of invertible heads, (3) the polynomial /
/// transcendental solver when available.  Candidates are **not** verified
/// here; every caller substitutes them back (or verifies the final answer).
pub(crate) fn isolate(eq: ExprId, v: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let eq = simp(eq, pool);
    if !contains(eq, v, pool) {
        return Vec::new();
    }
    let mut out = Vec::new();
    // 1. Linear in v.
    if let Some(dv) = d(eq, v, pool) {
        if !contains(dv, v, pool) && !is_zero(dv, pool) {
            let at0 = simp(subs1(eq, v, pool.integer(0), pool), pool);
            let sol = simp(neg(div(at0, dv, pool), pool), pool);
            if !contains(sol, v, pool) {
                out.push(sol);
                return out;
            }
        }
    }
    // 2. Peel.
    let mut stack = vec![(eq, pool.integer(0), 0usize)];
    while let Some((lhs, rhs, depth)) = stack.pop() {
        if depth > 40 {
            continue;
        }
        if lhs == v {
            let s = simp(rhs, pool);
            if !out.contains(&s) {
                out.push(s);
            }
            continue;
        }
        let data = pool.get(lhs);
        match data {
            ExprData::Add(args) => {
                let (with, without): (Vec<ExprId>, Vec<ExprId>) =
                    args.iter().partition(|&&a| contains(a, v, pool));
                if with.len() == 1 {
                    let r = sub(rhs, pool.add(without), pool);
                    stack.push((with[0], simp(r, pool), depth + 1));
                } else {
                    // Several terms: try linearity of the combined form.
                    let comb = simp(sub(lhs, rhs, pool), pool);
                    if let Some(dv) = d(comb, v, pool) {
                        if !contains(dv, v, pool) && !is_zero(dv, pool) {
                            let at0 = simp(subs1(comb, v, pool.integer(0), pool), pool);
                            let s = simp(neg(div(at0, dv, pool), pool), pool);
                            if !contains(s, v, pool) && !out.contains(&s) {
                                out.push(s);
                            }
                        }
                    }
                }
            }
            ExprData::Mul(args) => {
                let (with, without): (Vec<ExprId>, Vec<ExprId>) =
                    args.iter().partition(|&&a| contains(a, v, pool));
                if with.len() == 1 {
                    let r = div(rhs, pool.mul(without), pool);
                    stack.push((with[0], simp(r, pool), depth + 1));
                } else if rhs == pool.integer(0) {
                    // A product vanishes when one factor does.
                    for w in with {
                        stack.push((w, rhs, depth + 1));
                    }
                }
            }
            ExprData::Pow { base, exp } => {
                let in_base = contains(base, v, pool);
                let in_exp = contains(exp, v, pool);
                if in_base && !in_exp {
                    if let Some(q) = as_rational(exp, pool) {
                        if q == 0 {
                            continue;
                        }
                        let inv = pool.rational(q.denom().clone(), q.numer().clone());
                        let root = pool.pow(rhs, inv);
                        stack.push((base, simp(root, pool), depth + 1));
                        // Even integer power: the negative root as well.
                        if q.denom() == &1u32 && q.numer().is_even() {
                            stack.push((base, simp(neg(root, pool), pool), depth + 1));
                        }
                    } else {
                        let inv = recip(exp, pool);
                        stack.push((base, simp(pool.pow(rhs, inv), pool), depth + 1));
                    }
                } else if in_exp && !in_base {
                    let r = div(
                        pool.func("log", vec![rhs]),
                        pool.func("log", vec![base]),
                        pool,
                    );
                    stack.push((exp, simp(r, pool), depth + 1));
                }
            }
            ExprData::Func { name, args } if args.len() == 1 => {
                let inv = match name.as_str() {
                    "exp" => Some("log"),
                    "log" => Some("exp"),
                    "sin" => Some("asin"),
                    "cos" => Some("acos"),
                    "tan" => Some("atan"),
                    "asin" => Some("sin"),
                    "acos" => Some("cos"),
                    "atan" => Some("tan"),
                    "sinh" => Some("asinh"),
                    "cosh" => Some("acosh"),
                    "tanh" => Some("atanh"),
                    "asinh" => Some("sinh"),
                    "acosh" => Some("cosh"),
                    "atanh" => Some("tanh"),
                    "sqrt" => None,
                    _ => None,
                };
                if let Some(inv) = inv {
                    stack.push((args[0], simp(pool.func(inv, vec![rhs]), pool), depth + 1));
                } else if name == "sqrt" {
                    let r = pool.pow(rhs, pool.integer(2));
                    stack.push((args[0], simp(r, pool), depth + 1));
                }
            }
            _ => {}
        }
    }
    out.retain(|&s| !contains(s, v, pool));
    if !out.is_empty() {
        return out;
    }
    // 3. The solver, when built.
    #[cfg(feature = "groebner")]
    {
        out.extend(solver_isolate(eq, v, pool));
    }
    out
}

#[cfg(feature = "groebner")]
fn solver_isolate(eq: ExprId, v: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    use crate::solver::transcendental::{solve_transcendental, TranscendentalOutcome};
    let mut out = Vec::new();
    if let TranscendentalOutcome::Solved(vals) = solve_transcendental(eq, v, pool) {
        let _ = crate::solver::take_solve_side_conditions();
        // A periodic family carries an integer index `k`; the k = 0 member is
        // enough for a local inverse.
        for s in vals {
            out.push(s);
        }
        if !out.is_empty() {
            return out;
        }
    }
    // Polynomial in v (rational in everything else): only when it really is
    // a polynomial equation, and with a size guard.
    if pool.depth(eq) > 40 {
        return out;
    }
    if let Ok(crate::solver::SolutionSet::Finite(sols)) =
        crate::solver::solve_polynomial_system(vec![eq], vec![v], pool)
    {
        for s in sols {
            if let Some(&val) = s.first() {
                out.push(simp(val, pool));
            }
        }
    }
    let _ = crate::solver::take_solve_side_conditions();
    out
}

/// The first candidate of [`isolate`] that is free of `v`, preferring ones
/// without `RootOf`, Lambert-W or a periodic index.
pub(crate) fn isolate_one(eq: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let c = isolate(eq, v, pool);
    c.into_iter().min_by_key(|&s| pool.depth(s))
}

// ---------------------------------------------------------------------------
// Quadrature
// ---------------------------------------------------------------------------

/// `∫ e dv` (no constant) through the verified integrator; `None` on any
/// refusal.  A free-of-`v` integrand is done directly.
pub(crate) fn integrate(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let e = simp(e, pool);
    if e == pool.integer(0) {
        return Some(e);
    }
    if !contains(e, v, pool) {
        return Some(simp(pool.mul(vec![e, v]), pool));
    }
    if has_undefined(e, pool) {
        return None;
    }
    if let Ok(r) = crate::integrate::integrate(e, v, pool) {
        let val = r.value;
        if !names_in(&[val], pool).contains("Integral") {
            return Some(simp(val, pool));
        }
    }
    // Term by term, and `∫ c·(a v + b)^n dv` directly (the integrator can
    // refuse a power of a linear form with a symbolic constant whose sign it
    // cannot place; every caller verifies the final answer anyway).
    if let ExprData::Add(xs) = pool.get(e) {
        let mut parts = Vec::new();
        for x in xs {
            parts.push(integrate(x, v, pool)?);
        }
        return Some(simp(pool.add(parts), pool));
    }
    let (coef, core) = match pool.get(e) {
        ExprData::Mul(xs) => {
            let (with, without): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&a| contains(a, v, pool));
            if with.len() != 1 {
                return None;
            }
            (pool.mul(without), with[0])
        }
        _ => (pool.integer(1), e),
    };
    let (base, n) = match pool.get(core) {
        ExprData::Pow { base, exp } if !contains(exp, v, pool) => (base, exp),
        _ if core == v => (v, pool.integer(1)),
        _ => return None,
    };
    let a = d(base, v, pool)?;
    if contains(a, v, pool) || is_zero(a, pool) {
        return None;
    }
    let n1 = simp(pool.add(vec![n, pool.integer(1)]), pool);
    if is_zero(n1, pool) {
        let r = div(pool.func("log", vec![base]), a, pool);
        return Some(simp(pool.mul(vec![coef, r]), pool));
    }
    let r = div(pool.pow(base, n1), pool.mul(vec![a, n1]), pool);
    Some(simp(pool.mul(vec![coef, r]), pool))
}

// ---------------------------------------------------------------------------
// Numeric probes (choice only, never certification)
// ---------------------------------------------------------------------------

/// Pseudo-random but deterministic sample values for `syms` (index `k`
/// chooses the point).
pub(crate) fn sample_env(syms: &[ExprId], k: usize) -> HashMap<ExprId, f64> {
    let mut env = HashMap::new();
    for (i, &s) in syms.iter().enumerate() {
        // Distinct irrational-ish values in (0.3, 1.9).
        let t = ((i as f64 + 1.0) * 0.618_033_988_7 + (k as f64 + 1.0) * 0.414_213_562_3).fract();
        env.insert(s, 0.3 + 1.6 * t);
    }
    env
}

pub(crate) fn eval_at(e: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<f64> {
    crate::jit::eval_interp(e, env, pool).filter(|v| v.is_finite())
}

/// Numeric rank of a matrix of expressions, as the maximum over a few sample
/// points (a nonzero minor at one point shows the functions are independent).
pub(crate) fn numeric_rank(rows: &[Vec<ExprId>], syms: &[ExprId], pool: &ExprPool) -> usize {
    let mut best = 0;
    for k in 0..3 {
        let env = sample_env(syms, k);
        let mut m: Vec<Vec<f64>> = Vec::new();
        let mut ok = true;
        for r in rows {
            let mut row = Vec::new();
            for &e in r {
                match eval_at(e, &env, pool) {
                    Some(v) => row.push(v),
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                break;
            }
            m.push(row);
        }
        if !ok {
            continue;
        }
        best = best.max(rank_f64(m));
    }
    best
}

fn rank_f64(mut m: Vec<Vec<f64>>) -> usize {
    let rows = m.len();
    if rows == 0 {
        return 0;
    }
    let cols = m[0].len();
    let scale = m
        .iter()
        .flatten()
        .fold(0.0f64, |a, &b| a.max(b.abs()))
        .max(1e-300);
    let mut rank = 0;
    let mut r = 0;
    for c in 0..cols {
        if r >= rows {
            break;
        }
        let (piv, val) = (r..rows)
            .map(|i| (i, m[i][c].abs()))
            .fold((r, -1.0), |a, b| if b.1 > a.1 { b } else { a });
        if val <= 1e-9 * scale {
            continue;
        }
        m.swap(r, piv);
        for i in 0..rows {
            if i != r {
                let f = m[i][c] / m[r][c];
                for j in c..cols {
                    m[i][j] -= f * m[r][j];
                }
            }
        }
        r += 1;
        rank += 1;
    }
    rank
}

/// Free symbols of `exprs` (for sampling), in a stable order.
pub(crate) fn free_symbols(exprs: &[ExprId], pool: &ExprPool) -> Vec<ExprId> {
    let mut s = BTreeSet::new();
    for &e in exprs {
        s.extend(symbols(e, pool));
    }
    s.into_iter().collect()
}
