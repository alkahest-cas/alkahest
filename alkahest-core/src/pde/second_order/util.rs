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

/// `e^e`, with `exp(c log f + …)` rewritten to `f^c · exp(…)`.
pub(crate) fn exp(e: ExprId, pool: &ExprPool) -> ExprId {
    let s = simp(e, pool);
    if s == pool.integer(0) {
        return pool.integer(1);
    }
    simp(exp_of_logs(s, pool), pool)
}

/// Node count, for choosing the smaller of equivalent forms.
pub(crate) fn size(e: ExprId, pool: &ExprPool) -> u64 {
    crate::simplify::full::expr_size(e, pool)
}

/// The smallest of a few value-preserving normal forms.
fn smallest(cands: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    cands
        .into_iter()
        .min_by_key(|&c| size(c, pool))
        .expect("non-empty")
}

/// `p / q` with a complex `q = a + b·I` rationalised: `p (a − bI)/(a² + b²)`.
pub(crate) fn cdiv(p: ExprId, q: ExprId, pool: &ExprPool) -> ExprId {
    let plain = div(p, q, pool);
    if !mentions_i(q, pool) {
        return plain;
    }
    let i = pool.imaginary_unit();
    let q = crate::simplify::engine::simplify_expanded(q, pool).value;
    let a = simp(subs1(q, i, pool.integer(0), pool), pool);
    let b = match d(q, i, pool) {
        Ok(b) => b,
        Err(_) => return plain,
    };
    if mentions_i(a, pool) || mentions_i(b, pool) {
        return plain;
    }
    let num = pool.mul(vec![p, sub(a, pool.mul(vec![b, i]), pool)]);
    let den = pool.add(vec![pool.pow(a, pool.integer(2)), pool.pow(b, pool.integer(2))]);
    let r = tidy(div(num, den, pool), pool);
    smallest(vec![plain, r], pool)
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
            // (a·b)^r → a^r b^r also for rational r: only used to *propose*
            // a separation, which the final substitution check vets.
            let is_num =
                pool.with(exp, |d| matches!(d, ExprData::Integer(_) | ExprData::Rational(_)));
            let inner = pool.with(base, |d| match d {
                ExprData::Mul(xs) => Some(xs.clone()),
                _ => None,
            });
            match (is_num, inner) {
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
    if let Some((out, rad)) = sqrt_parts(e, pool) {
        return simp(
            pool.mul(vec![out, pool.pow(rad, pool.rational(1, 2))]),
            pool,
        );
    }
    simp(pool.pow(e, pool.rational(1, 2)), pool)
}

/// `e = out² · rad` with the square part of a polynomial `e` pulled out
/// (FLINT factorisation); `None` when nothing comes out.
fn sqrt_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let e = simp(e, pool);
    let vars: Vec<ExprId> = symbols_of(e, pool).into_iter().collect();
    if vars.is_empty() || vars.iter().any(|&v| pool.is_imaginary_unit(v)) {
        return None;
    }
    let mp = crate::poly::multipoly::MultiPoly::from_symbolic(e, vars, pool).ok()?;
    let f = crate::poly::factor::factor_multivariate_z(&mp).ok()?;
    let unit = f.unit.clone();
    let neg_unit = unit < 0;
    let ua = unit.abs();
    // largest square dividing the unit (trial division is enough here)
    let mut sq_out = rug::Integer::from(1);
    let mut rest = ua.clone();
    let mut p = rug::Integer::from(2);
    while p.clone() * p.clone() <= rest && p < 10_000 {
        let pp = p.clone() * p.clone();
        while rest.is_divisible(&pp) {
            rest /= &pp;
            sq_out *= &p;
        }
        p += 1;
    }
    let mut outs = vec![pool.integer(sq_out.clone())];
    let mut rads = vec![pool.integer(if neg_unit { -rest } else { rest })];
    let mut pulled = sq_out != 1;
    for (q, m) in &f.factors {
        let qe = q.to_expr(pool);
        if *m >= 2 {
            pulled = true;
            outs.push(pool.pow(qe, pool.integer(*m as i64 / 2)));
        }
        if m % 2 == 1 {
            rads.push(qe);
        }
    }
    pulled.then(|| (simp(pool.mul(outs), pool), simp(pool.mul(rads), pool)))
}

/// Replace `I` by `−I` (complex conjugation when everything else is real).
pub(crate) fn conj(e: ExprId, pool: &ExprPool) -> ExprId {
    let i = pool.imaginary_unit();
    let e = tidy(e, pool);
    tidy(subs1(e, i, neg(i, pool), pool), pool)
}

pub(crate) fn mentions_i(e: ExprId, pool: &ExprPool) -> bool {
    symbols_of(e, pool)
        .iter()
        .any(|&s| pool.is_imaginary_unit(s))
}

/// A cosmetic rewrite that preserves the value everywhere it is defined:
/// `(b^r)^n → b^{rn}` for integer `n`, and
/// `exp(c₁ log f₁ + … + rest) → f₁^{c₁} ⋯ exp(rest)` (principal branches:
/// `exp(c log f) = f^c` by definition), then two simplifier passes.
pub(crate) fn tidy(e: ExprId, pool: &ExprPool) -> ExprId {
    let mut memo = HashMap::new();
    let r = rewrite(e, pool, &mut memo);
    let a = simp(simp(r, pool), pool);
    let mut cands = vec![a];
    let ex = simp(crate::simplify::engine::simplify_expanded(a, pool).value, pool);
    cands.push(ex);
    if has_trig(a, pool) {
        cands.push(simp(crate::simplify::engine::simplify_trig(a, pool).value, pool));
    }
    smallest(cands, pool)
}

fn has_trig(e: ExprId, pool: &ExprPool) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if pool.with(n, |d| matches!(d, ExprData::Func { name, .. } if name == "sin" || name == "cos")) {
            return true;
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    false
}

fn rewrite(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
    if let Some(&r) = memo.get(&e) {
        return r;
    }
    let out = match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| rewrite(x, pool, memo)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| rewrite(x, pool, memo)).collect()),
        ExprData::Pow { base, exp } => {
            let b = rewrite(base, pool, memo);
            let x = rewrite(exp, pool, memo);
            let outer_int = pool.with(x, |d| matches!(d, ExprData::Integer(_)));
            if outer_int && pool.is_imaginary_unit(b) {
                let n = pool.with(x, |d| match d {
                    ExprData::Integer(n) => n.0.mod_u(4),
                    _ => 0,
                });
                let i = pool.imaginary_unit();
                let r = match n {
                    0 => pool.integer(1),
                    1 => i,
                    2 => pool.integer(-1),
                    _ => pool.mul(vec![pool.integer(-1), i]),
                };
                memo.insert(e, r);
                return r;
            }
            let inner = pool.with(b, |d| match d {
                ExprData::Pow { base, exp } => Some((*base, *exp)),
                _ => None,
            });
            match (outer_int, inner) {
                (true, Some((bb, r))) => {
                    let ex = simp(pool.mul(vec![r, x]), pool);
                    if ex == pool.integer(1) {
                        bb
                    } else {
                        pool.pow(bb, ex)
                    }
                }
                _ => pool.pow(b, x),
            }
        }
        ExprData::Func { name, args } => {
            let args: Vec<ExprId> = args.iter().map(|&a| rewrite(a, pool, memo)).collect();
            if name == "exp" && args.len() == 1 {
                exp_of_logs(args[0], pool)
            } else {
                pool.func(name, args)
            }
        }
        _ => e,
    };
    memo.insert(e, out);
    out
}

fn exp_of_logs(arg: ExprId, pool: &ExprPool) -> ExprId {
    let arg = simp(arg, pool);
    let terms = match pool.get(arg) {
        ExprData::Add(ts) => ts,
        _ => vec![arg],
    };
    let is_num =
        |z: ExprId| pool.with(z, |d| matches!(d, ExprData::Integer(_) | ExprData::Rational(_)));
    let mut powers = Vec::new();
    let mut rest = Vec::new();
    for t in terms {
        // c · log(f) with c rational
        let (c, body) = match pool.get(t) {
            ExprData::Mul(fs) if fs.len() == 2 && is_num(fs[0]) => (fs[0], fs[1]),
            ExprData::Mul(fs) if fs.len() == 2 && is_num(fs[1]) => (fs[1], fs[0]),
            _ => (pool.integer(1), t),
        };
        let log_arg = pool.with(body, |d| match d {
            ExprData::Func { name, args } if name == "log" && args.len() == 1 => Some(args[0]),
            _ => None,
        });
        match log_arg {
            Some(f) => powers.push(if c == pool.integer(1) {
                f
            } else {
                pool.pow(f, c)
            }),
            None => rest.push(t),
        }
    }
    if powers.is_empty() {
        return pool.func("exp", vec![arg]);
    }
    if !rest.is_empty() {
        powers.push(pool.func("exp", vec![pool.add(rest)]));
    }
    pool.mul(powers)
}

fn has_log(e: ExprId, pool: &ExprPool) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if pool.with(n, |d| matches!(d, ExprData::Func { name, .. } if name == "log")) {
            return true;
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    false
}

/// `Σ c_k(x, y) · A_k + r(x, y)` with the coefficients of the arbitrary-
/// function atoms `A_k` (`F1(y)`, `Derivative(F2(x), x)`, …) each brought to
/// one cancelled rational function.  `None` when the expression is not
/// linear in the atoms or nothing changes.
pub(crate) fn collect_arbitrary(e: ExprId, vars: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut atoms: Vec<ExprId> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if crate::diff::undefined::view(n, pool).is_some() {
            if !atoms.contains(&n) {
                atoms.push(n);
            }
            continue;
        }
        super::super::jet::push_children(n, pool, &mut stack);
    }
    if atoms.is_empty() {
        return None;
    }
    let syms: Vec<ExprId> = (0..atoms.len())
        .map(|k| pool.symbol(format!("__alkahest_pde2_atom{k}"), Domain::Real))
        .collect();
    let map: HashMap<ExprId, ExprId> = atoms.iter().copied().zip(syms.iter().copied()).collect();
    let s = crate::kernel::subs::subs(e, &map, pool);
    let norm = |c: ExprId| -> ExprId {
        let c = simp(c, pool);
        match crate::poly::cancel::cancel(c, vars.to_vec(), pool) {
            Ok(r) => smallest(vec![c, simp(r, pool)], pool),
            Err(_) => c,
        }
    };
    let mut terms = Vec::new();
    for (&a, &sy) in atoms.iter().zip(&syms) {
        let c = d(s, sy, pool).ok()?;
        if !free_of_all(c, &syms, pool) {
            return None;
        }
        let c = norm(c);
        if c != pool.integer(0) {
            terms.push(pool.mul(vec![c, a]));
        }
    }
    let zero_map: HashMap<ExprId, ExprId> = syms.iter().map(|&s| (s, pool.integer(0))).collect();
    let rest = norm(crate::kernel::subs::subs(s, &zero_map, pool));
    if rest != pool.integer(0) {
        terms.push(rest);
    }
    let out = simp(pool.add(terms), pool);
    (out != e).then_some(out)
}

/// Prefer a first integral free of logarithms: `e^φ` when that is.
pub(crate) fn nicer_integral(phi: ExprId, pool: &ExprPool) -> ExprId {
    let phi = tidy(phi, pool);
    if !has_log(phi, pool) {
        return phi;
    }
    let e = tidy(pool.func("exp", vec![phi]), pool);
    if has_log(e, pool) {
        phi
    } else {
        e
    }
}
