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
    let s = if names_in(&[s], pool).contains(crate::specfun::hyper::HYPER) {
        simp(crate::pde::verify::contract_hyper(s, pool), pool)
    } else {
        s
    };
    if zero_test(s, pool, None).status == ZeroStatus::ProvedZero {
        return true;
    }
    // The same expression with powers merged (identities only; see
    // `normalize_powers`), which the normal form does not do for symbolic
    // exponents.
    let n = crate::pde::verify::normalize_powers(s, pool);
    n != s && zero_test(n, pool, None).status == ZeroStatus::ProvedZero
}

/// [`is_zero`] for expressions that may contain `Integral` nodes: the
/// integrals are alpha-normalised, the expression distributed, and
/// `Σ cᵢ ∫ fᵢ` combined into one integral before the zero test (as in
/// verification).
pub(crate) fn is_zero_with_integrals(e: ExprId, pool: &ExprPool) -> bool {
    if !crate::pde::integral::has_integral(e, pool) {
        return is_zero(e, pool);
    }
    if is_zero(e, pool) {
        return true;
    }
    let a = crate::pde::integral::alpha_normalize_integrals(e, pool);
    let x = crate::simplify::engine::simplify_expanded(a, pool).value;
    let c = crate::pde::integral::combine_integrals(x, pool);
    is_zero(c, pool)
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
                        // 0^(1/n) is 0 for a positive power (the simplifier
                        // leaves it alone).
                        let root = if rhs == pool.integer(0) && q > 0 {
                            rhs
                        } else {
                            root_of(rhs, inv, pool)
                        };
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

/// `rhs^inv`, distributing over a product and multiplying exponents of
/// powers (`(a²)^(1/2) → a`).  A branch choice; callers verify (the other
/// sign of an even root is pushed separately).
fn root_of(rhs: ExprId, inv: ExprId, pool: &ExprPool) -> ExprId {
    let one = |f: ExprId| match pool.get(f) {
        ExprData::Pow { base, exp } if is_number(exp, pool) => {
            pool.pow(base, simp(pool.mul(vec![exp, inv]), pool))
        }
        _ => pool.pow(f, inv),
    };
    let r = match pool.get(rhs) {
        ExprData::Mul(xs) => pool.mul(xs.into_iter().map(one).collect()),
        _ => one(rhs),
    };
    simp(r, pool)
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
    // A rational equation is solved through its numerator (roots of the
    // denominator are not solutions, and every caller verifies anyway).
    let num = match crate::poly::cancel::together_parts(eq, vec![v], pool) {
        Ok((n, _)) => n,
        Err(_) => eq,
    };
    for target in [eq, num] {
        if let Ok(crate::solver::SolutionSet::Finite(sols)) =
            crate::solver::solve_polynomial_system(vec![target], vec![v], pool)
        {
            for s in sols {
                if let Some(&val) = s.first() {
                    let val = simp(val, pool);
                    if !out.contains(&val) {
                        out.push(val);
                    }
                }
            }
        }
        let _ = crate::solver::take_solve_side_conditions();
        if !out.is_empty() || num == eq {
            break;
        }
    }
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
    integrate_depth(e, v, 2, pool)
}

fn integrate_depth(e: ExprId, v: ExprId, depth: u32, pool: &ExprPool) -> Option<ExprId> {
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
    if let Some(r) = integrate_core(e, v, pool) {
        return Some(r);
    }
    // The same integrand cleaned (`exp(c·log a) → a^c`, function arguments
    // expanded), then expanded into a sum of terms.
    let t = super::tidy(e, pool);
    if t != e {
        if let Some(r) = integrate_core(t, v, pool) {
            return Some(r);
        }
    }
    if let Some(r) = special_quadrature(t, v, pool) {
        return Some(r);
    }
    if depth > 0 && pool.depth(t) < 14 {
        let ex = crate::simplify::engine::simplify_expanded(t, pool).value;
        if ex != t && ex != e {
            if let ExprData::Add(_) = pool.get(ex) {
                if let Some(r) = integrate_depth(ex, v, depth - 1, pool) {
                    return Some(r);
                }
            }
        }
        if let Some(r) = exp_substitution(t, v, pool) {
            return Some(r);
        }
        let tc = rewrite_tan(t, pool);
        if tc != t {
            if let Some(r) = integrate_core(simp(tc, pool), v, pool) {
                return Some(r);
            }
        }
    }
    // Term by term, and `∫ c·(a v + b)^n dv` directly (the integrator can
    // refuse a power of a linear form with a symbolic constant whose sign it
    // cannot place; every caller verifies the final answer anyway).
    if let ExprData::Add(xs) = pool.get(e) {
        let mut parts = Vec::new();
        for x in xs {
            parts.push(integrate_depth(x, v, depth, pool)?);
        }
        return Some(simp(pool.add(parts), pool));
    }
    if depth > 0 {
        if let Some(r) = by_parts(e, v, depth, pool) {
            return Some(r);
        }
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

/// `∫ e dv` in closed form when [`integrate`] finds one, otherwise the
/// unevaluated `Integral(e[v := t], t, lo, v)` (see [`mod@crate::pde::integral`]):
/// answers containing it are verified exactly through the fundamental
/// theorem of calculus.  The lower limit is the first of `1, 0, 2, −1` where
/// the integrand evaluates to a finite number at a sample point.
pub(crate) fn quad(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if let Some(r) = integrate(e, v, pool) {
        return Some(r);
    }
    unevaluated(e, v, pool)
}

pub(crate) fn unevaluated(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let e = simp(e, pool);
    if !contains(e, v, pool) {
        // Nothing to leave unevaluated.
        return Some(simp(pool.mul(vec![e, v]), pool));
    }
    let t = crate::pde::integral::fresh_dummy("_t", &[e, v], pool);
    let body = subs1(e, v, t, pool);
    let inst = instantiate_undefined(body, pool);
    let syms: Vec<ExprId> = free_symbols(&[inst], pool)
        .into_iter()
        .filter(|&s| s != t)
        .collect();
    // A lower limit where the integrand is finite on the segment to the
    // sample value of v, at two sample points of the parameters.
    let mut lo = pool.integer(1);
    'cands: for cand in [1i64, 0, 2, -1] {
        for k in 0..2 {
            let env0 = sample_env(&syms, k);
            let vv = sample_env(&[v], k)[&v];
            for i in 0..=8 {
                let tv = cand as f64 + (vv - cand as f64) * (i as f64) / 8.0;
                let mut env = env0.clone();
                env.insert(t, tv);
                if eval_at(inst, &env, pool).is_none() {
                    continue 'cands;
                }
            }
        }
        lo = pool.integer(cand);
        break;
    }
    crate::pde::integral::integral(body, t, lo, v, pool).ok()
}

/// Replace every applied undefined function (and its derivatives) by a fixed
/// generic elementary function, for numeric probes only.
pub(crate) fn instantiate_undefined(e: ExprId, pool: &ExprPool) -> ExprId {
    let mut apps: Vec<(String, usize)> = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some(vw) = crate::diff::undefined::view(n, pool) {
            if !apps
                .iter()
                .any(|(nm, a)| nm == &vw.name && *a == vw.args.len())
            {
                apps.push((vw.name.clone(), vw.args.len()));
            }
            stack.extend(vw.args);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    let mut cur = e;
    for (k, (name, arity)) in apps.into_iter().enumerate() {
        let xi: Vec<ExprId> = (0..arity)
            .map(|j| pool.symbol(format!("__alkahest_inst{j}"), Domain::Real))
            .collect();
        let lin = pool.add(
            xi.iter()
                .enumerate()
                .map(|(j, &x)| pool.mul(vec![pool.rational(j as i64 + 3 + k as i64, 7), x]))
                .collect(),
        );
        let body = pool.add(vec![
            pool.func("sin", vec![lin]),
            pool.func("exp", vec![pool.mul(vec![pool.rational(1, 5), lin])]),
        ]);
        let pattern = pool.func(name.clone(), xi.clone());
        if let Ok(r) = crate::diff::undefined::replace_function(cur, pattern, body, pool) {
            cur = r;
        }
        // Applications the replacement did not reach (inside an `Integral`).
        cur = replace_apps(cur, &name, &xi, body, pool);
    }
    cur
}

fn replace_apps(e: ExprId, name: &str, params: &[ExprId], body: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Func { name: n, args } if n == name && args.len() == params.len() => {
            let args: Vec<ExprId> = args
                .iter()
                .map(|&a| replace_apps(a, name, params, body, pool))
                .collect();
            let m: HashMap<ExprId, ExprId> = params.iter().copied().zip(args).collect();
            subs_map(body, &m, pool)
        }
        ExprData::Func { name: n, args } => pool.func(
            n,
            args.iter()
                .map(|&a| replace_apps(a, name, params, body, pool))
                .collect(),
        ),
        ExprData::Add(xs) => pool.add(
            xs.iter()
                .map(|&a| replace_apps(a, name, params, body, pool))
                .collect(),
        ),
        ExprData::Mul(xs) => pool.mul(
            xs.iter()
                .map(|&a| replace_apps(a, name, params, body, pool))
                .collect(),
        ),
        ExprData::Pow { base, exp } => pool.pow(
            replace_apps(base, name, params, body, pool),
            replace_apps(exp, name, params, body, pool),
        ),
        _ => e,
    }
}

// ---------------------------------------------------------------------------
// Special-function quadratures with symbolic parameters
// ---------------------------------------------------------------------------

/// Closed forms the integrator declines when a parameter's sign or a power is
/// symbolic.  Every one is a *candidate*: the callers verify the PDE answer
/// (or the first integral) exactly, so a form the zero test cannot confirm is
/// discarded, never returned.
///
/// * `∫ P(v) e^{αv²+βv+γ}`, `P` a polynomial of degree ≤ 4, `α` symbolic:
///   `√π/(2√α) e^{γ−β²/4α} erfi(√α (v + β/2α))` and the moment recursion;
/// * `∫ vᵐ e^{a vᵖ + γ}` (symbolic `m`, `p`):
///   `−e^γ v^{m+1} (−a vᵖ)^{−(m+1)/p} Γ((m+1)/p, −a vᵖ)/p`;
/// * `∫ e^{βv + c e^{λv} + γ}` through `w = e^{λv}`, which is the previous
///   form in `w`;
/// * `∫ Wᵏ` for `W′ = λ(1 + σW²)`, `σ = ±1` (`W = tan, cot, tanh, coth` of a
///   linear argument), symbolic `k`:
///   `W^{k+1}/(λ(k+1)) ₂F₁(1, (k+1)/2; (k+3)/2; −σW²)`.
fn special_quadrature(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    gaussian_quadrature(e, v, pool)
        .or_else(|| incomplete_gamma_quadrature(e, v, pool))
        .or_else(|| exp_of_exp_quadrature(e, v, pool))
        .or_else(|| tangent_power_quadrature(e, v, pool))
}

/// `(rest, Σ args)` of a product whose exponential factors in `v` are
/// `exp(argᵢ)`; `None` when there is no such factor.
fn split_exp(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let xs = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    };
    let mut rest = Vec::new();
    let mut args = Vec::new();
    for x in xs {
        match pool.get(x) {
            ExprData::Func { name, args: a }
                if name == "exp" && a.len() == 1 && contains(a[0], v, pool) =>
            {
                args.push(a[0])
            }
            _ => rest.push(x),
        }
    }
    if args.is_empty() {
        return None;
    }
    Some((simp(pool.mul(rest), pool), simp(pool.add(args), pool)))
}

/// Taylor coefficients `[p₀, …, p_d]` of a polynomial `p` in `v` of degree
/// `d ≤ max`; `None` when `p` is not one.
fn poly_coeffs(p: ExprId, v: ExprId, max: usize, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let zero = pool.integer(0);
    let mut out = Vec::new();
    let mut cur = p;
    let mut fact = rug::Integer::from(1);
    for k in 0..=max + 1 {
        if is_zero(cur, pool) {
            return Some(out);
        }
        if k == max + 1 {
            return None;
        }
        let c = simp(subs1(cur, v, zero, pool), pool);
        out.push(simp(
            pool.mul(vec![c, pool.rational(rug::Integer::from(1), fact.clone())]),
            pool,
        ));
        cur = d(cur, v, pool)?;
        fact *= k as u32 + 1;
    }
    None
}

fn gaussian_quadrature(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (p, q) = split_exp(e, v, pool)?;
    if contains(p, v, pool) && has_undefined(p, pool) {
        return None;
    }
    let q1 = d(q, v, pool)?;
    let q2 = d(q1, v, pool)?;
    if contains(q2, v, pool) || is_zero(q2, pool) {
        return None;
    }
    let zero = pool.integer(0);
    let alpha = simp(pool.mul(vec![pool.rational(1, 2), q2]), pool);
    let beta = simp(subs1(q1, v, zero, pool), pool);
    let gamma = simp(subs1(q, v, zero, pool), pool);
    let coeffs = poly_coeffs(p, v, 4, pool)?;
    if coeffs.iter().any(|&c| contains(c, v, pool)) {
        return None;
    }
    let pi = pool.symbol("pi", Domain::Real);
    let s = pool.func("sqrt", vec![alpha]);
    let two_a = pool.mul(vec![pool.integer(2), alpha]);
    let shift = pool.add(vec![v, div(beta, two_a, pool)]);
    let pref = pool.mul(vec![
        pool.func("sqrt", vec![pi]),
        recip(pool.mul(vec![pool.integer(2), s]), pool),
        pool.func(
            "exp",
            vec![sub(
                gamma,
                div(
                    pool.pow(beta, pool.integer(2)),
                    pool.mul(vec![pool.integer(4), alpha]),
                    pool,
                ),
                pool,
            )],
        ),
    ]);
    let i0 = pool.mul(vec![
        pref,
        pool.func("erfi", vec![pool.mul(vec![s, shift])]),
    ]);
    let eq = pool.func("exp", vec![q]);
    let mut ik: Vec<ExprId> = vec![i0];
    for k in 1..coeffs.len() {
        let mut num = vec![pool.mul(vec![pool.pow(v, pool.integer(k as i64 - 1)), eq])];
        if k >= 2 {
            num.push(neg(
                pool.mul(vec![pool.integer(k as i64 - 1), ik[k - 2]]),
                pool,
            ));
        }
        num.push(neg(pool.mul(vec![beta, ik[k - 1]]), pool));
        ik.push(div(pool.add(num), two_a, pool));
    }
    let terms: Vec<ExprId> = coeffs
        .iter()
        .zip(&ik)
        .map(|(&c, &i)| pool.mul(vec![c, i]))
        .collect();
    Some(simp(pool.add(terms), pool))
}

/// `(K, m)` with `r = K·vᵐ`, `K` and `m` free of `v` (by the logarithmic
/// derivative `v r′/r`).
fn monomial(r: ExprId, v: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if !contains(r, v, pool) {
        return Some((r, pool.integer(0)));
    }
    if let Some(km) = monomial_syntactic(r, v, pool) {
        return Some(km);
    }
    let m = simp(div(pool.mul(vec![v, d(r, v, pool)?]), r, pool), pool);
    if contains(m, v, pool) {
        return None;
    }
    let k = simp(div(r, pool.pow(v, m), pool), pool);
    if contains(k, v, pool) {
        let k = simp(super::tidy(k, pool), pool);
        if contains(k, v, pool) {
            return None;
        }
        return Some((k, m));
    }
    Some((k, m))
}

/// `(K, m)` read off the factors of `r`: factors free of `v`, `v`, `vᵉ`, and
/// integer powers of such products (`(a vⁿ)⁻¹ = a⁻¹ v⁻ⁿ`).
fn monomial_syntactic(r: ExprId, v: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    fn walk(
        x: ExprId,
        k: ExprId,
        v: ExprId,
        pool: &ExprPool,
        ks: &mut Vec<ExprId>,
        ms: &mut Vec<ExprId>,
    ) -> Option<()> {
        // x^k with k an integer (or 1 at the top).
        if !contains(x, v, pool) {
            ks.push(pool.pow(x, k));
            return Some(());
        }
        if x == v {
            ms.push(k);
            return Some(());
        }
        match pool.get(x) {
            ExprData::Mul(ys) => {
                for y in ys {
                    walk(y, k, v, pool, ks, ms)?;
                }
                Some(())
            }
            ExprData::Pow { base, exp } if !contains(exp, v, pool) => {
                if base == v {
                    ms.push(pool.mul(vec![exp, k]));
                    return Some(());
                }
                // Only an integer power distributes over a product.
                as_rational(exp, pool).filter(|q| q.denom() == &1u32)?;
                walk(base, simp(pool.mul(vec![exp, k]), pool), v, pool, ks, ms)
            }
            _ => None,
        }
    }
    let mut ks = Vec::new();
    let mut ms = Vec::new();
    walk(r, pool.integer(1), v, pool, &mut ks, &mut ms)?;
    Some((simp(pool.mul(ks), pool), simp(pool.add(ms), pool)))
}

/// `q = a vᵖ + γ` read off the terms of `q`: one term `a·vᵖ` (or `a·v`), the
/// others free of `v`.
fn power_term(q: ExprId, v: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let q = crate::simplify::engine::simplify_expanded(q, pool).value;
    let terms = match pool.get(q) {
        ExprData::Add(xs) => xs,
        _ => vec![q],
    };
    let (with, without): (Vec<ExprId>, Vec<ExprId>) =
        terms.into_iter().partition(|&t| contains(t, v, pool));
    if with.len() != 1 {
        return None;
    }
    let (a, p) = monomial_syntactic(with[0], v, pool)?;
    if is_zero(p, pool) {
        return None;
    }
    Some((a, p, simp(pool.add(without), pool)))
}

/// `∫ wᵐ e^{a w} dw = −w^{m+1}(−a w)^{−(m+1)} Γ(m+1, −a w)` generalised to
/// `∫ K vᵐ e^{a vᵖ + γ}`.
fn upper_gamma_form(
    k: ExprId,
    m: ExprId,
    a: ExprId,
    p: ExprId,
    gamma: ExprId,
    v: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let m1 = pool.add(vec![m, pool.integer(1)]);
    let s = div(m1, p, pool);
    let z = neg(pool.mul(vec![a, pool.pow(v, p)]), pool);
    simp(
        pool.mul(vec![
            pool.integer(-1),
            k,
            pool.func("exp", vec![gamma]),
            pool.pow(v, m1),
            pool.pow(z, neg(s, pool)),
            pool.func("uppergamma", vec![s, z]),
            recip(p, pool),
        ]),
        pool,
    )
}

fn incomplete_gamma_quadrature(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (r, q) = split_exp(e, v, pool)?;
    if has_undefined(r, pool) {
        return None;
    }
    let (k, m) = monomial(r, v, pool)?;
    if let Some((a, p, gamma)) = power_term(q, v, pool) {
        return Some(upper_gamma_form(k, m, a, p, gamma, v, pool));
    }
    // q = a vᵖ + γ: p = 1 + v q″/q′.
    let q1 = d(q, v, pool)?;
    if is_zero(q1, pool) {
        return None;
    }
    let q2 = d(q1, v, pool)?;
    let p = simp(
        pool.add(vec![pool.integer(1), div(pool.mul(vec![v, q2]), q1, pool)]),
        pool,
    );
    if contains(p, v, pool) || is_zero(p, pool) {
        return None;
    }
    // A linear exponent with an integer power is the integrator's (it
    // declined for a reason), a quadratic one the Gaussian rule's.
    let a = simp(
        super::tidy(
            div(
                pool.mul(vec![q1, pool.pow(v, sub(pool.integer(1), p, pool))]),
                p,
                pool,
            ),
            pool,
        ),
        pool,
    );
    if contains(a, v, pool) || is_zero(a, pool) {
        return None;
    }
    let gamma = simp(sub(q, pool.mul(vec![a, pool.pow(v, p)]), pool), pool);
    if contains(gamma, v, pool) {
        return None;
    }
    Some(upper_gamma_form(k, m, a, p, gamma, v, pool))
}

fn exp_of_exp_quadrature(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (r, s) = split_exp(e, v, pool)?;
    if contains(r, v, pool) {
        return None;
    }
    // s = βv + γ + T, T = c e^{λv + μ}: λ = T‴/T″ on s.
    let s1 = d(s, v, pool)?;
    let s2 = d(s1, v, pool)?;
    if is_zero(s2, pool) {
        return None;
    }
    let s3 = d(s2, v, pool)?;
    let lam = simp(div(s3, s2, pool), pool);
    if contains(lam, v, pool) || is_zero(lam, pool) {
        return None;
    }
    let t = simp(div(s2, pool.pow(lam, pool.integer(2)), pool), pool);
    let beta = simp(sub(s1, pool.mul(vec![lam, t]), pool), pool);
    if contains(beta, v, pool) {
        return None;
    }
    let gamma = simp(
        sub(s, pool.add(vec![pool.mul(vec![beta, v]), t]), pool),
        pool,
    );
    if contains(gamma, v, pool) {
        return None;
    }
    let ev = pool.func("exp", vec![pool.mul(vec![lam, v])]);
    let c = simp(super::tidy(div(t, ev, pool), pool), pool);
    if contains(c, v, pool) {
        return None;
    }
    // ∫ r e^γ e^{βv} e^{c w} dv = (r e^γ/λ) ∫ w^{β/λ − 1} e^{c w} dw at w = e^{λv}.
    let mut taken = names_in(&[e, v], pool);
    let w = fresh_symbol("__pde_w", &mut taken, pool);
    let m = simp(sub(div(beta, lam, pool), pool.integer(1), pool), pool);
    let iw = upper_gamma_form(div(r, lam, pool), m, c, pool.integer(1), gamma, w, pool);
    Some(simp(subs1(iw, w, ev, pool), pool))
}

fn tangent_power_quadrature(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let xs = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    };
    let (with, without): (Vec<ExprId>, Vec<ExprId>) =
        xs.into_iter().partition(|&x| contains(x, v, pool));
    if with.len() != 1 {
        return None;
    }
    let c = pool.mul(without);
    // W^k, also (W^a)^b written as W^(ab) when a = −1 is not involved.
    let (w, k) = match pool.get(with[0]) {
        ExprData::Pow { base, exp } if !contains(exp, v, pool) => match pool.get(base) {
            // (Wⁿ)⁻¹ = W⁻ⁿ exactly.
            ExprData::Pow { base: b2, exp: e2 }
                if !contains(e2, v, pool) && exp == pool.integer(-1) =>
            {
                (b2, simp(neg(e2, pool), pool))
            }
            _ => (base, exp),
        },
        _ => return None,
    };
    if as_rational(k, pool).is_some() {
        // A literal power is the integrator's business.
        return None;
    }
    let wp = d(w, v, pool)?;
    let w2 = pool.pow(w, pool.integer(2));
    for sigma in [1i64, -1] {
        let den = pool.add(vec![
            pool.integer(1),
            pool.mul(vec![pool.integer(sigma), w2]),
        ]);
        let lam = simp(super::tidy(div(wp, den, pool), pool), pool);
        if contains(lam, v, pool) || is_zero(lam, pool) {
            continue;
        }
        let k1 = simp(pool.add(vec![k, pool.integer(1)]), pool);
        if is_zero(k1, pool) {
            return None;
        }
        let b = simp(pool.mul(vec![pool.rational(1, 2), k1]), pool);
        let b1 = simp(pool.add(vec![b, pool.integer(1)]), pool);
        let z = simp(pool.mul(vec![pool.integer(-sigma), w2]), pool);
        let f = crate::specfun::hyper::raw_node(&[pool.integer(1), b], &[b1], z, pool);
        return Some(simp(
            pool.mul(vec![
                c,
                pool.pow(w, k1),
                recip(pool.mul(vec![lam, k1]), pool),
                f,
            ]),
            pool,
        ));
    }
    None
}

fn integrate_core(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Principal-branch logs: first integrals are defined up to a constant, and
    // the algebra downstream (solving for parameters, exponentiating) wants
    // `log x`, not `log|x|`.
    let r = {
        let _principal = crate::integrate::real_log::PrincipalLogs::enter();
        crate::integrate::integrate(e, v, pool).ok()?
    };
    let val = r.value;
    if names_in(&[val], pool).contains("Integral") {
        return None;
    }
    Some(simp(val, pool))
}

/// `∫ R(e^{λv}) dv` with `t = e^{λv}`: hyperbolic functions are rewritten
/// through `exp`, every exponential of `v` must be `e^{(rλ)v + c}` with a
/// rational `r`, and the integrand becomes `R(t)/(λ t)` in `t`.  The
/// integrator often declines the original form when `λ` is symbolic
/// (`∫ dy/sinh(μy)`); the answer is verified by every caller.
fn exp_substitution(e: ExprId, v: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let e = rewrite_hyperbolic(e, v, pool);
    let e = super::tidy(e, pool);
    // Exponential nodes in v.
    let mut nodes = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![e];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        let is_exp = pool.with(
            n,
            |d| matches!(d, ExprData::Func { name, args } if name == "exp" && args.len() == 1),
        );
        if is_exp && contains(n, v, pool) {
            nodes.push(n);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if nodes.is_empty() {
        return None;
    }
    let arg_of = |n: ExprId| match pool.get(n) {
        ExprData::Func { args, .. } => args[0],
        _ => n,
    };
    let lam = d(arg_of(nodes[0]), v, pool)?;
    if contains(lam, v, pool) || is_zero(lam, pool) {
        return None;
    }
    let mut taken = names_in(&[e], pool);
    let t = loop {
        let name = format!("__pde_t{}", taken.len());
        if taken.insert(name.clone()) {
            break pool.symbol(name, crate::kernel::Domain::Positive);
        }
    };
    let mut m = HashMap::new();
    for &n in &nodes {
        let a = arg_of(n);
        let ai = d(a, v, pool)?;
        let r = simp(div(ai, lam, pool), pool);
        let q = as_rational(r, pool)?;
        if q.denom() != &1u32 || q.numer().clone().abs() > 6 {
            return None;
        }
        let rest = simp(sub(a, pool.mul(vec![ai, v]), pool), pool);
        if contains(rest, v, pool) {
            return None;
        }
        let tp = pool.pow(t, pool.integer(q.numer().to_i64()?));
        let repl = if rest == pool.integer(0) {
            tp
        } else {
            pool.mul(vec![pool.func("exp", vec![rest]), tp])
        };
        m.insert(n, repl);
    }
    let et = simp(subs_map(e, &m, pool), pool);
    if contains(et, v, pool) {
        return None;
    }
    let integrand = div(et, pool.mul(vec![lam, t]), pool);
    let it = integrate_core(simp(integrand, pool), t, pool)?;
    let back = pool.func("exp", vec![pool.mul(vec![lam, v])]);
    Some(simp(subs1(it, t, back, pool), pool))
}

/// `tan → sin/cos` (so `1/tan` is `cos/sin`), for the integrator.
fn rewrite_tan(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| rewrite_tan(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| rewrite_tan(x, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(rewrite_tan(base, pool), exp),
        ExprData::Func { name, args } if name == "tan" && args.len() == 1 => div(
            pool.func("sin", vec![args[0]]),
            pool.func("cos", vec![args[0]]),
            pool,
        ),
        ExprData::Func { name, args } if name == "tanh" && args.len() == 1 => div(
            pool.func("sinh", vec![args[0]]),
            pool.func("cosh", vec![args[0]]),
            pool,
        ),
        _ => e,
    }
}

fn rewrite_hyperbolic(e: ExprId, v: ExprId, pool: &ExprPool) -> ExprId {
    if !contains(e, v, pool) {
        return e;
    }
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| rewrite_hyperbolic(x, v, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| rewrite_hyperbolic(x, v, pool)).collect()),
        ExprData::Pow { base, exp } => pool.pow(rewrite_hyperbolic(base, v, pool), exp),
        ExprData::Func { name, args } if args.len() == 1 => {
            let z = rewrite_hyperbolic(args[0], v, pool);
            let ep = pool.func("exp", vec![z]);
            let em = pool.func("exp", vec![neg(z, pool)]);
            let half = pool.rational(1, 2);
            match name.as_str() {
                "sinh" => pool.mul(vec![half, sub(ep, em, pool)]),
                "cosh" => pool.mul(vec![half, pool.add(vec![ep, em])]),
                "tanh" => div(sub(ep, em, pool), pool.add(vec![ep, em]), pool),
                _ => pool.func(name, vec![z]),
            }
        }
        _ => e,
    }
}

/// `∫ A·L dv = (∫A)·L − ∫ (∫A)·L′ dv` for a logarithmic or inverse
/// trigonometric factor `L` (`∫ eˣ log x dx = eˣ log x − Ei(x)`).
fn by_parts(e: ExprId, v: ExprId, depth: u32, pool: &ExprPool) -> Option<ExprId> {
    let xs = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => return None,
    };
    for (i, &l) in xs.iter().enumerate() {
        let is_l = pool.with(l, |d| {
            matches!(d, ExprData::Func { name, .. }
                if matches!(name.as_str(), "log" | "atan" | "asin" | "acos"))
        });
        if !is_l || !contains(l, v, pool) {
            continue;
        }
        let rest: Vec<ExprId> = xs
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &x)| x)
            .collect();
        let a = pool.mul(rest);
        let Some(ga) = integrate_depth(a, v, depth - 1, pool) else {
            continue;
        };
        let Some(dl) = d(l, v, pool) else {
            continue;
        };
        let Some(second) = integrate_depth(pool.mul(vec![ga, dl]), v, depth - 1, pool) else {
            continue;
        };
        return Some(simp(sub(pool.mul(vec![ga, l]), second, pool), pool));
    }
    None
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
    // Arbitrary functions get generic instances (the rank at one instance
    // bounds the generic rank from below).
    let rows: Vec<Vec<ExprId>> = rows
        .iter()
        .map(|r| r.iter().map(|&e| instantiate_undefined(e, pool)).collect())
        .collect();
    let rows = &rows[..];
    let mut syms = syms.to_vec();
    for r in rows {
        for s in free_symbols(r, pool) {
            if !syms.contains(&s) {
                syms.push(s);
            }
        }
    }
    let syms = &syms[..];
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
                let pivot_row = m[r].clone();
                for (cell, pr) in m[i].iter_mut().zip(pivot_row).skip(c) {
                    *cell -= f * pr;
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

#[cfg(test)]
mod special_quadrature_tests {
    use super::*;

    /// `F′ − f` is proved zero for each rule's antiderivative.
    #[test]
    fn special_quadratures_differentiate_back() {
        let pool = ExprPool::new();
        let mut syms = HashMap::new();
        let t = pool.symbol("t", Domain::Real);
        syms.insert("t".to_string(), t);
        let mut failures = Vec::new();
        for (src, head) in [
            ("exp(a*t^2)", "erfi"),
            ("(s + b*t^2 + c*t)*exp(-k*t - a*t^2/2)", "erfi"),
            ("exp(a*t^3)", "uppergamma"),
            // Γ(1, z) = e^{−z}: the rule's answer simplifies to elementary.
            ("t^n*exp(a*t^(n+1))", "exp"),
            ("t^m*exp(-c*t^(k+1)/(k+1))", "uppergamma"),
            ("exp(b*t + 2*a*exp(lamda*t)/lamda)", "uppergamma"),
            ("tan(lamda*t)^k", "hyper"),
            ("tanh(lamda*t)^k", "hyper"),
            ("(tan(mu*t)^n)^(-1)", "hyper"),
        ] {
            let f = simp(crate::parse::parse(src, &pool, &mut syms).unwrap(), &pool);
            let Some(r) = special_quadrature(super::super::tidy(f, &pool), t, &pool) else {
                failures.push(format!("{src}: no special quadrature"));
                continue;
            };
            if !names_in(&[r], &pool).contains(head) {
                failures.push(format!("{src}: {} has no {head}", pool.display(r)));
                continue;
            }
            let dr = d(r, t, &pool).unwrap();
            if !is_zero(sub(dr, f, &pool), &pool) {
                failures.push(format!(
                    "{src}: d/dt[{}] − f = {} not proved zero",
                    pool.display(r),
                    pool.display(simp(sub(dr, f, &pool), &pool))
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
}
