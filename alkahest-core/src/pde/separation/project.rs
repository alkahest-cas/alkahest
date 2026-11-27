//! Coefficients by orthogonal projection: `c = ∫ W·f·Φ / ‖Φ‖²`.
//!
//! Each one-dimensional integral is done by an antiderivative that is
//! **checked exactly** (`F' − integrand` proved zero), then the
//! fundamental theorem of calculus; a value at an end that is singular is
//! taken as a one-sided limit.  `integrate` is tried first, then a small
//! table for `x^k J_ν(a x)` (the Bessel-coefficient integrals of disk and
//! cylinder problems), then term-by-term.  When no antiderivative closes, the
//! coefficient is kept as an unevaluated `Integral(f, s, lo, hi)` — the
//! reserved head of the PDE layer, with a **fresh dummy variable** so that no
//! substitution can capture it.
//!
//! `exceptional_indices` finds the integer index values where a closed-form
//! coefficient's denominator vanishes (`1/(n² − 1)` for data `sin x` on
//! `(0, π)`); those members are recomputed separately.

use std::collections::HashMap;

use crate::kernel::subs::mentions_var;
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::ZeroStatus;

use super::super::jet::{push_children, symbols_of};
use super::super::sturm_liouville::{d, simp, subs1, zstatus};
use super::modes::{div, neg, sub};

/// The reserved head of an unevaluated definite integral.
pub const INTEGRAL: &str = "Integral";

/// `Integral(body[var := s], s, lo, hi)` with a fresh dummy `s`.
pub(crate) fn integral_placeholder(
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> ExprId {
    let base = match pool.get(var) {
        ExprData::Symbol { name, .. } => format!("{name}_"),
        _ => "s_".into(),
    };
    let dummy =
        super::super::sturm_liouville::fresh_symbol(&base, Domain::Real, &[body, lo, hi], pool);
    let b = subs1(body, var, dummy, pool);
    pool.func(INTEGRAL, vec![b, dummy, lo, hi])
}

/// `(body, var, lo, hi)` of an `Integral` placeholder.
pub fn integral_parts(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId, ExprId)> {
    pool.with(e, |dd| match dd {
        ExprData::Func { name, args } if name == INTEGRAL && args.len() == 4 => {
            Some((args[0], args[1], args[2], args[3]))
        }
        _ => None,
    })
}

/// Whether `e` contains an `Integral` placeholder.
pub fn has_integral(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if integral_parts(n, pool).is_some() {
            return true;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

/// A division by zero or an infinity in `e`.
pub(crate) fn has_singular_value(e: ExprId, pool: &ExprPool) -> bool {
    if crate::simplify::find_division_by_zero(e, pool).is_some() {
        return true;
    }
    let oo = pool.pos_infinity();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if n == oo {
            return true;
        }
        let bad = pool.with(n, |dd| match dd {
            ExprData::Symbol { name, .. } => {
                matches!(
                    name.as_str(),
                    "zoo" | "nan" | "NaN" | "ComplexInfinity" | "oo"
                )
            }
            ExprData::Func { name, args } => {
                (name == "log" && args.len() == 1 && args[0] == pool.integer(0))
                    || (name == "Undefined")
            }
            _ => false,
        });
        if bad {
            return true;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

fn proved(e: ExprId, pool: &ExprPool) -> bool {
    if zstatus(e, pool) == ZeroStatus::ProvedZero {
        return true;
    }
    let ex = expand_exp(super::separate::expanded(e, pool), pool);
    zstatus(ex, pool) == ZeroStatus::ProvedZero
}

/// An antiderivative of `body` in `var` whose derivative is proved equal to
/// `body`; `None` when none is found.
pub(crate) fn antiderivative(body: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let body = simp(body, pool);
    if !mentions_var(body, var, pool) {
        return Some(pool.mul(vec![body, var]));
    }
    // Pull out the factors free of the variable (also keeps a Piecewise in
    // another variable away from the integrator).
    if let ExprData::Mul(xs) = pool.get(body) {
        let (free, dep): (Vec<ExprId>, Vec<ExprId>) =
            xs.iter().partition(|&&f| !mentions_var(f, var, pool));
        if !free.is_empty() && !dep.is_empty() {
            let inner = antiderivative(pool.mul(dep), var, pool)?;
            return Some(simp(pool.mul(vec![pool.mul(free), inner]), pool));
        }
    }
    let check = |f: ExprId| -> bool {
        let df = match d(f, var, pool) {
            Ok(x) => x,
            Err(_) => return false,
        };
        proved(sub(df, body, pool), pool)
    };
    if let Some(f) = bessel_power(body, var, pool) {
        if check(f) {
            return Some(f);
        }
    }
    if let Ok(r) = crate::integrate::integrate(body, var, pool) {
        let f = r.value;
        if check(f) {
            return Some(f);
        }
    }
    let lin = linearize_trig(body, pool);
    if lin != body {
        if let Some(f) = antiderivative_terms(lin, var, pool) {
            if check(f) {
                return Some(f);
            }
        }
    }
    let f = antiderivative_terms(body, var, pool)?;
    check(f).then_some(f)
}

/// Term-by-term antiderivative after expansion (unchecked).
fn antiderivative_terms(body: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ex = super::separate::expanded(body, pool);
    let terms = match pool.get(ex) {
        ExprData::Add(xs) if xs.len() > 1 => xs,
        _ => {
            if let Some(f) = bessel_power(ex, var, pool) {
                return Some(f);
            }
            return crate::integrate::integrate(ex, var, pool)
                .ok()
                .map(|r| r.value);
        }
    };
    let mut parts = Vec::new();
    for t in terms {
        let f = if !mentions_var(t, var, pool) {
            pool.mul(vec![t, var])
        } else if let Some(f) = bessel_power(t, var, pool) {
            f
        } else {
            crate::integrate::integrate(t, var, pool).ok()?.value
        };
        parts.push(f);
    }
    Some(simp(pool.add(parts), pool))
}

/// The order and argument of a Bessel `J` node: `bessel_j(ν, z)`,
/// `bessel_j0(z)`, `bessel_j1(z)`.
fn bessel_node(e: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    pool.with(e, |dd| match dd {
        ExprData::Func { name, args } => match (name.as_str(), args.len()) {
            ("bessel_j", 2) => Some((args[0], args[1])),
            ("bessel_j0", 1) => Some((pool.integer(0), args[0])),
            ("bessel_j1", 1) => Some((pool.integer(1), args[0])),
            _ => None,
        },
        _ => None,
    })
}

/// `∫ c x^k J_ν(a x) dx` summed over the terms of `body`, when every term
/// has `k − ν − 1` an even non-negative integer (the integration-by-parts
/// recursion `∫ x^k J_ν = x^k J_{ν+1}/a − (k−ν−1)/a ∫ x^{k−1} J_{ν+1}`
/// terminates at `∫ x^{ν+1} J_ν = x^{ν+1} J_{ν+1}/a`).
fn bessel_power(body: ExprId, x: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let ex = super::separate::expanded(body, pool);
    let terms = match pool.get(ex) {
        ExprData::Add(xs) => xs,
        _ => vec![ex],
    };
    let mut out = Vec::new();
    for t in terms {
        let factors = match pool.get(t) {
            ExprData::Mul(xs) => xs,
            _ => vec![t],
        };
        let mut c = Vec::new();
        let mut k = pool.integer(0);
        let mut bes: Option<(ExprId, ExprId)> = None;
        for f in factors {
            if !mentions_var(f, x, pool) {
                c.push(f);
                continue;
            }
            if f == x {
                k = simp(pool.add(vec![k, pool.integer(1)]), pool);
                continue;
            }
            if let ExprData::Pow { base, exp } = pool.get(f) {
                if base == x && !mentions_var(exp, x, pool) {
                    k = simp(pool.add(vec![k, exp]), pool);
                    continue;
                }
            }
            if let Some((nu, z)) = bessel_node(f, pool) {
                if bes.is_some() || mentions_var(nu, x, pool) {
                    return None;
                }
                let a = simp(d(z, x, pool).ok()?, pool);
                if mentions_var(a, x, pool)
                    || simp(sub(z, pool.mul(vec![a, x]), pool), pool) != pool.integer(0)
                {
                    return None;
                }
                bes = Some((nu, a));
                continue;
            }
            return None;
        }
        let (nu, a) = bes?;
        let j = simp(sub(sub(k, nu, pool), pool.integer(1), pool), pool);
        let jv = pool.with(j, |dd| match dd {
            ExprData::Integer(n) => n.0.to_i64(),
            _ => None,
        })?;
        if jv < 0 || jv % 2 != 0 || jv > 40 {
            return None;
        }
        // Recursion: I(k, ν) = x^k J_{ν+1}/a − (k−ν−1)/a I(k−1, ν+1).
        let mut acc = Vec::new();
        let mut coef = pool.integer(1);
        let (mut kk, mut nn) = (k, nu);
        loop {
            let jj = simp(sub(sub(kk, nn, pool), pool.integer(1), pool), pool);
            let nn1 = simp(pool.add(vec![nn, pool.integer(1)]), pool);
            let term = pool.mul(vec![
                coef,
                pool.pow(x, kk),
                pool.func("bessel_j", vec![nn1, pool.mul(vec![a, x])]),
                pool.pow(a, pool.integer(-1)),
            ]);
            acc.push(term);
            if jj == pool.integer(0) {
                break;
            }
            coef = simp(pool.mul(vec![coef, neg(div(jj, a, pool), pool)]), pool);
            kk = simp(sub(kk, pool.integer(1), pool), pool);
            nn = nn1;
        }
        out.push(pool.mul(vec![pool.mul(c), pool.add(acc)]));
    }
    Some(simp(pool.add(out), pool))
}

/// `F(at)`, or the one-sided limit when the substitution is singular.
fn value_at(
    f: ExprId,
    var: ExprId,
    at: ExprId,
    from_above: bool,
    pool: &ExprPool,
) -> Option<ExprId> {
    let v = simp(subs1(f, var, at, pool), pool);
    if !has_singular_value(v, pool) {
        return Some(v);
    }
    let dir = if from_above {
        crate::calculus::limits::LimitDirection::Plus
    } else {
        crate::calculus::limits::LimitDirection::Minus
    };
    let l = crate::calculus::limits::limit(f, var, at, dir, pool).ok()?;
    let l = simp(l, pool);
    (!has_singular_value(l, pool)).then_some(l)
}

thread_local! {
    /// Denominators of the antiderivatives used since the last
    /// [`take_denominators`]: the FTC value is invalid where one vanishes, so
    /// the caller expands those index values separately.
    static DENOMS: std::cell::RefCell<Vec<ExprId>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The denominators recorded by [`definite`] since the last call.
pub(crate) fn take_denominators() -> Vec<ExprId> {
    DENOMS.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

fn record_denominators(f: ExprId, pool: &ExprPool) {
    let mut found = Vec::new();
    let mut stack = vec![f];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(n) {
            let negative = pool.with(exp, |dd| match dd {
                ExprData::Integer(k) => k.0 < 0,
                ExprData::Rational(q) => q.0 < 0,
                _ => false,
            });
            if negative
                && symbols_of(base, pool).iter().any(|&s| {
                    pool.with(s, |dd| {
                        matches!(
                            dd,
                            ExprData::Symbol {
                                domain: Domain::Integer,
                                ..
                            }
                        )
                    })
                })
            {
                found.push(base);
            }
        }
        push_children(n, pool, &mut stack);
    }
    if !found.is_empty() {
        DENOMS.with(|d| d.borrow_mut().extend(found));
    }
}

/// `∫_{lo}^{hi} body d(var)` by an exactly checked antiderivative.
pub(crate) fn definite(
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let Some(f) = antiderivative(body, var, pool) else {
        return definite_piecewise(body, var, lo, hi, pool);
    };
    record_denominators(f, pool);
    let a = value_at(f, var, lo, true, pool)?;
    let b = value_at(f, var, hi, false, pool)?;
    Some(reduce_integer_trig(simp(sub(b, a, pool), pool), pool))
}

/// The iterated integral `∫…∫ body dvars` (first variable innermost):
/// `(value, closed)`.  Variables that do not close are kept as nested
/// `Integral` placeholders.
pub(crate) fn iterated(
    body: ExprId,
    vars: &[(ExprId, ExprId, ExprId)],
    pool: &ExprPool,
) -> (ExprId, bool) {
    let mut cur = simp(body, pool);
    for (i, &(v, lo, hi)) in vars.iter().enumerate() {
        match definite(cur, v, lo, hi, pool) {
            Some(val) => cur = val,
            None => {
                // Term by term: close what closes, keep the rest.
                let ex = super::separate::expanded(cur, pool);
                if let ExprData::Add(xs) = pool.get(ex) {
                    let mut parts = Vec::new();
                    let mut closed = true;
                    for t in xs {
                        let (v2, c2) = iterated(t, &vars[i..], pool);
                        closed &= c2;
                        parts.push(v2);
                    }
                    return (simp(pool.add(parts), pool), closed);
                }
                let mut e = integral_placeholder(cur, v, lo, hi, pool);
                for &(w, lo2, hi2) in &vars[i + 1..] {
                    e = integral_placeholder(e, w, lo2, hi2, pool);
                }
                return (e, false);
            }
        }
    }
    (cur, true)
}

/// Integer values `≥ first` of `index` at which a denominator of `e`
/// vanishes, and predicates `base ≠ 0` for denominators whose integer zeros
/// cannot be decided (symbolic parameters).
pub(crate) fn exceptional_indices(
    e: ExprId,
    index: ExprId,
    first: i64,
    others: &[ExprId],
    pool: &ExprPool,
) -> (Vec<i64>, Vec<ExprId>) {
    let mut bases = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(n) {
            let negative = pool.with(exp, |dd| match dd {
                ExprData::Integer(k) => k.0 < 0,
                ExprData::Rational(q) => q.0 < 0,
                _ => false,
            });
            if negative && mentions_var(base, index, pool) {
                bases.push(base);
            }
        }
        if integral_parts(n, pool).is_some() {
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    bases.sort();
    bases.dedup();
    let mut out = Vec::new();
    let mut side = Vec::new();
    for b in bases {
        if others.iter().any(|&o| mentions_var(b, o, pool)) {
            side.push(ne0(b, pool));
            continue;
        }
        if super::super::root_family::root_family_parts(b, pool).is_some()
            || contains_root_family(b, pool)
        {
            continue;
        }
        let free: Vec<ExprId> = symbols_of(b, pool)
            .into_iter()
            .filter(|&s| s != index && !is_constant_symbol(s, pool))
            .collect();
        // Polynomial coefficients (Taylor at 0), up to degree 8.
        let mut coeffs = Vec::new();
        let mut cur = b;
        let mut fact = 1.0f64;
        let mut poly = false;
        for k in 0..=9 {
            if k > 0 {
                fact *= k as f64;
            }
            let c0 = simp(subs1(cur, index, pool.integer(0), pool), pool);
            coeffs.push((c0, fact));
            cur = match d(cur, index, pool) {
                Ok(x) => simp(x, pool),
                Err(_) => break,
            };
            if cur == pool.integer(0) {
                poly = true;
                break;
            }
        }
        let numeric: Option<Vec<f64>> = coeffs
            .iter()
            .map(|&(c, f)| {
                crate::eval::eval_f64(c, pool, &HashMap::new())
                    .ok()
                    .map(|v| v / f)
            })
            .collect();
        // Cauchy's bound on the integer roots, with the parameters at two
        // positive samples when there are any (the exact scan below decides;
        // the bound only sets how far it goes).
        let cauchy = |cs: &[f64]| -> f64 {
            let lead = cs
                .iter()
                .rev()
                .find(|c| c.abs() > 1e-300)
                .copied()
                .unwrap_or(1.0);
            1.0 + cs.iter().map(|c| (c / lead).abs()).fold(0.0, f64::max)
        };
        let sampled: Vec<Option<Vec<f64>>> = [1.3f64, 2.9]
            .iter()
            .map(|&v| {
                let env: HashMap<ExprId, f64> = free.iter().map(|&s| (s, v)).collect();
                coeffs
                    .iter()
                    .map(|&(c, f)| {
                        crate::jit::eval_interp_real(c, &env, pool)
                            .ok()
                            .filter(|x| x.is_finite())
                            .map(|x| x / f)
                    })
                    .collect()
            })
            .collect();
        let scan_to: i64 = match (poly, free.is_empty(), &numeric) {
            (true, true, Some(cs)) => (cauchy(cs).ceil() as i64).clamp(first + 1, first + 10_000),
            (true, false, _) => {
                let b1 = sampled
                    .iter()
                    .flatten()
                    .map(|cs| cauchy(cs))
                    .fold(0.0, f64::max);
                ((b1.ceil() as i64) + 60).clamp(first + 60, first + 10_000)
            }
            _ => first + 60,
        };
        let mut found = 0usize;
        for k in first..=scan_to {
            let v = simp(subs1(b, index, pool.integer(k), pool), pool);
            if v == pool.integer(0)
                || crate::simplify::zero::zero_test(v, pool, None).status == ZeroStatus::ProvedZero
            {
                out.push(k);
                found += 1;
            }
        }
        // A parametric denominator may still vanish at an integer index for
        // special parameter values: state that as a side condition unless
        // every root of the polynomial was found.
        let degree = coeffs.len().saturating_sub(1);
        if !free.is_empty() && !(poly && found >= degree && degree > 0) {
            side.push(ne0(b, pool));
        }
    }
    out.sort();
    out.dedup();
    (out, side)
}

fn ne0(b: ExprId, pool: &ExprPool) -> ExprId {
    pool.predicate(
        crate::kernel::expr::PredicateKind::Ne,
        vec![b, pool.integer(0)],
    )
}

fn is_constant_symbol(s: ExprId, pool: &ExprPool) -> bool {
    pool.with(s, |dd| match dd {
        ExprData::Symbol { name, .. } => crate::eval::symbols::constant_value_f64(name).is_some(),
        _ => false,
    })
}

pub(crate) fn contains_root_family(e: ExprId, pool: &ExprPool) -> bool {
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if super::super::root_family::root_family_parts(n, pool).is_some() {
            return true;
        }
        push_children(n, pool, &mut stack);
    }
    false
}

/// Product-to-sum: every product of `sin` / `cos` factors (small integer
/// powers expanded) becomes a sum of single `sin` / `cos` terms.
pub(crate) fn linearize_trig(e: ExprId, pool: &ExprPool) -> ExprId {
    #[derive(Clone, Copy, PartialEq)]
    enum K {
        One,
        Sin,
        Cos,
    }
    let half = pool.rational(1, 2);
    let ex = super::separate::expanded(e, pool);
    let terms = match pool.get(ex) {
        ExprData::Add(xs) => xs,
        _ => vec![ex],
    };
    let mut out = Vec::new();
    let mut changed = false;
    for t in terms {
        let factors = match pool.get(t) {
            ExprData::Mul(xs) => xs,
            _ => vec![t],
        };
        let mut trig: Vec<(K, ExprId)> = Vec::new();
        let mut rest = Vec::new();
        for f in factors {
            let one = |g: ExprId| -> Option<(K, ExprId)> {
                pool.with(g, |dd| match dd {
                    ExprData::Func { name, args } if args.len() == 1 && name == "sin" => {
                        Some((K::Sin, args[0]))
                    }
                    ExprData::Func { name, args } if args.len() == 1 && name == "cos" => {
                        Some((K::Cos, args[0]))
                    }
                    _ => None,
                })
            };
            if let Some(k) = one(f) {
                trig.push(k);
                continue;
            }
            if let ExprData::Pow { base, exp } = pool.get(f) {
                let n = pool.with(exp, |dd| match dd {
                    ExprData::Integer(k) => k.0.to_i64(),
                    _ => None,
                });
                if let (Some(k), Some(n)) = (one(base), n) {
                    if (1..=8).contains(&n) {
                        for _ in 0..n {
                            trig.push(k);
                        }
                        continue;
                    }
                }
            }
            rest.push(f);
        }
        if trig.len() < 2 || trig.len() > 10 {
            out.push(t);
            continue;
        }
        changed = true;
        let mut acc: Vec<(ExprId, K, ExprId)> = vec![(pool.integer(1), K::One, pool.integer(0))];
        for (k, a) in trig {
            let mut next = Vec::new();
            for &(c, kb, b) in &acc {
                let s = |x: ExprId, y: ExprId| pool.add(vec![x, y]);
                let m =
                    |x: ExprId, y: ExprId| pool.add(vec![x, pool.mul(vec![pool.integer(-1), y])]);
                let hc = pool.mul(vec![half, c]);
                let nhc = pool.mul(vec![pool.integer(-1), half, c]);
                match (kb, k) {
                    (K::One, kk) => next.push((c, kk, a)),
                    (K::Sin, K::Sin) => {
                        next.push((hc, K::Cos, m(b, a)));
                        next.push((nhc, K::Cos, s(b, a)));
                    }
                    (K::Sin, K::Cos) => {
                        next.push((hc, K::Sin, s(b, a)));
                        next.push((hc, K::Sin, m(b, a)));
                    }
                    (K::Cos, K::Sin) => {
                        next.push((hc, K::Sin, s(b, a)));
                        next.push((nhc, K::Sin, m(b, a)));
                    }
                    (K::Cos, K::Cos) => {
                        next.push((hc, K::Cos, m(b, a)));
                        next.push((hc, K::Cos, s(b, a)));
                    }
                    (_, K::One) => next.push((c, kb, b)),
                }
            }
            acc = next;
        }
        let r = pool.mul(rest);
        for (c, k, a) in acc {
            let a = simp(a, pool);
            let f = match k {
                K::One => pool.integer(1),
                K::Sin => pool.func("sin", vec![a]),
                K::Cos => pool.func("cos", vec![a]),
            };
            out.push(pool.mul(vec![c, r, f]));
        }
    }
    if !changed {
        return e;
    }
    simp(pool.add(out), pool)
}

/// Whether `k` is integer-valued for integer values of its integer-domain
/// symbols: built from integers and such symbols by `+`, `·` and
/// non-negative integer powers.
fn integer_valued(k: ExprId, pool: &ExprPool) -> bool {
    match pool.get(k) {
        ExprData::Integer(_) => true,
        ExprData::Symbol { domain, .. } => domain == Domain::Integer,
        ExprData::Add(xs) | ExprData::Mul(xs) => xs.iter().all(|&x| integer_valued(x, pool)),
        ExprData::Pow { base, exp } => {
            integer_valued(base, pool)
                && pool.with(exp, |dd| matches!(dd, ExprData::Integer(n) if n.0 >= 0))
        }
        _ => false,
    }
}

/// `sin(kπ + c) = (−1)^k sin c`, `cos(kπ + c) = (−1)^k cos c` for an
/// integer-valued `k` (index symbols are integers).
pub(crate) fn reduce_integer_trig(e: ExprId, pool: &ExprPool) -> ExprId {
    let pi = pool.symbol("pi", Domain::Real);
    fn go(e: ExprId, pi: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let r = match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pi, pool, memo)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pi, pool, memo)).collect()),
            ExprData::Pow { base, exp } => {
                pool.pow(go(base, pi, pool, memo), go(exp, pi, pool, memo))
            }
            ExprData::Func { name, args }
                if args.len() == 1 && (name == "sin" || name == "cos") =>
            {
                let a = go(args[0], pi, pool, memo);
                let ex = super::separate::expanded(a, pool);
                let terms = match pool.get(ex) {
                    ExprData::Add(xs) => xs,
                    _ => vec![ex],
                };
                let mut ks = Vec::new();
                let mut rest = Vec::new();
                for t in terms {
                    let k = simp(pool.mul(vec![t, pool.pow(pi, pool.integer(-1))]), pool);
                    if !mentions_var(k, pi, pool) && integer_valued(k, pool) && k != pool.integer(0)
                    {
                        ks.push(k);
                    } else {
                        rest.push(t);
                    }
                }
                if ks.is_empty() {
                    pool.func(name, vec![a])
                } else {
                    let k = simp(pool.add(ks), pool);
                    let c = simp(pool.add(rest), pool);
                    let sign = pool.pow(pool.integer(-1), k);
                    pool.mul(vec![sign, pool.func(name, vec![c])])
                }
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, pi, pool, memo)).collect())
            }
            _ => e,
        };
        memo.insert(e, r);
        r
    }
    let r = go(e, pi, pool, &mut HashMap::new());
    if r == e {
        return e;
    }
    simp(r, pool)
}

/// At a zero `z` of `J_m` (the node `zero`), rewrite every `J_{m+k}(z)` with
/// integer `k` through the three-term recurrence in terms of `J_{m+1}(z)`
/// alone (`J_m(z) = 0`).  `e` unchanged when `zero` is not a Bessel node.
pub(crate) fn bessel_at_root(e: ExprId, zero: ExprId, pool: &ExprPool) -> ExprId {
    let Some((m, z)) = bessel_node(zero, pool) else {
        return e;
    };
    let mut nodes: Vec<(i64, ExprId)> = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let Some((o, zz)) = bessel_node(n, pool) {
            if zz == z {
                let k = simp(sub(o, m, pool), pool);
                if let Some(k) = pool.with(k, |dd| match dd {
                    ExprData::Integer(i) => i.0.to_i64(),
                    _ => None,
                }) {
                    if (-20..=20).contains(&k) {
                        nodes.push((k, n));
                    }
                }
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    if nodes.is_empty() {
        return e;
    }
    let ord = |k: i64| simp(pool.add(vec![m, pool.integer(k)]), pool);
    let j1 = simp(pool.func("bessel_j", vec![ord(1), z]), pool);
    let lo = nodes.iter().map(|n| n.0).min().unwrap_or(0).min(0);
    let hi = nodes.iter().map(|n| n.0).max().unwrap_or(1).max(1);
    let mut val: HashMap<i64, ExprId> = HashMap::new();
    val.insert(0, pool.integer(0));
    val.insert(1, j1);
    let two_k_over_z =
        |k: i64| pool.mul(vec![pool.integer(2), ord(k), pool.pow(z, pool.integer(-1))]);
    for k in 1..hi {
        let next = sub(
            pool.mul(vec![two_k_over_z(k), val[&k]]),
            val[&(k - 1)],
            pool,
        );
        val.insert(k + 1, simp(next, pool));
    }
    for k in (lo + 1..=0).rev() {
        let prev = sub(
            pool.mul(vec![two_k_over_z(k), val[&k]]),
            val[&(k + 1)],
            pool,
        );
        val.insert(k - 1, simp(prev, pool));
    }
    let map: HashMap<ExprId, ExprId> = nodes.iter().map(|&(k, n)| (n, val[&k])).collect();
    simp(super::separate::replace_nodes(e, &map, pool), pool)
}

/// `exp(a + b) → exp(a)·exp(b)` (arguments expanded), so that exponentials
/// written with different groupings of the same exponent compare equal.
pub(crate) fn expand_exp(e: ExprId, pool: &ExprPool) -> ExprId {
    fn go(e: ExprId, pool: &ExprPool, memo: &mut HashMap<ExprId, ExprId>) -> ExprId {
        if let Some(&r) = memo.get(&e) {
            return r;
        }
        let r = match pool.get(e) {
            ExprData::Add(xs) => pool.add(xs.iter().map(|&x| go(x, pool, memo)).collect()),
            ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| go(x, pool, memo)).collect()),
            ExprData::Pow { base, exp } => pool.pow(go(base, pool, memo), go(exp, pool, memo)),
            ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                let a = super::separate::expanded(go(args[0], pool, memo), pool);
                match pool.get(a) {
                    ExprData::Add(xs) => {
                        pool.mul(xs.iter().map(|&x| pool.func("exp", vec![x])).collect())
                    }
                    _ => pool.func("exp", vec![a]),
                }
            }
            ExprData::Func { name, args } => {
                pool.func(name, args.iter().map(|&x| go(x, pool, memo)).collect())
            }
            _ => e,
        };
        memo.insert(e, r);
        r
    }
    let r = go(e, pool, &mut HashMap::new());
    simp(r, pool)
}

/// `1^x → 1` and other cosmetic folds the simplifier leaves.
pub(crate) fn tidy(e: ExprId, pool: &ExprPool) -> ExprId {
    let one = pool.integer(1);
    let mut map = HashMap::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, .. } = pool.get(n) {
            if base == one {
                map.insert(n, one);
                continue;
            }
        }
        push_children(n, pool, &mut stack);
    }
    if map.is_empty() {
        return e;
    }
    simp(super::separate::replace_nodes(e, &map, pool), pool)
}

/// `zstatus`, retried after expanding products and exponentials when it
/// does not prove zero.
pub(crate) fn zero_status(e: ExprId, pool: &ExprPool) -> ZeroStatus {
    let st = zstatus(e, pool);
    if st == ZeroStatus::ProvedZero || st == ZeroStatus::ProvedNonzero {
        return st;
    }
    if proved(e, pool) {
        return ZeroStatus::ProvedZero;
    }
    st
}

/// `d/ds` of an expression that may contain `Integral` placeholders, by
/// Leibniz's rule: `d/ds ∫_{lo(s)}^{hi(s)} b(s, τ) dτ = b(s, hi)·hi' −
/// b(s, lo)·lo' + ∫ ∂b/∂s dτ`.
pub(crate) fn diff_with_integrals(e: ExprId, s: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // Outermost Integral nodes.
    let mut nodes = Vec::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if integral_parts(n, pool).is_some() {
            nodes.push(n);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if nodes.is_empty() {
        return d(e, s, pool).ok();
    }
    let mut fwd = HashMap::new();
    let mut back = HashMap::new();
    let mut leib = HashMap::new();
    for (k, &n) in nodes.iter().enumerate() {
        if !mentions_var(n, s, pool) {
            // Constant in s: an opaque symbol.
            let c = pool.symbol(format!("__leibniz_c{k}_{}", n.0), Domain::Real);
            fwd.insert(n, c);
            back.insert(c, n);
            continue;
        }
        let g = pool.func(format!("__leibniz_G{k}"), vec![s]);
        fwd.insert(n, g);
        back.insert(g, n);
        let (b, tau, lo, hi) = integral_parts(n, pool)?;
        let db = diff_with_integrals(b, s, pool)?;
        let dlo = d(lo, s, pool).ok()?;
        let dhi = d(hi, s, pool).ok()?;
        let val = pool.add(vec![
            pool.mul(vec![subs1(b, tau, hi, pool), dhi]),
            neg(pool.mul(vec![subs1(b, tau, lo, pool), dlo]), pool),
            pool.func(INTEGRAL, vec![simp(db, pool), tau, lo, hi]),
        ]);
        leib.insert(crate::diff::undefined::build(g, &[1], pool), val);
    }
    let e2 = super::separate::replace_nodes(e, &fwd, pool);
    let de = d(e2, s, pool).ok()?;
    let de = super::separate::replace_nodes(de, &leib, pool);
    Some(super::separate::replace_nodes(de, &back, pool))
}

/// Replace every `Integral` node by a fresh symbol (an opaque atom): an
/// identity proved with them independent holds for their actual values.
pub(crate) fn opaque_integrals(e: ExprId, pool: &ExprPool) -> ExprId {
    let mut map = HashMap::new();
    let mut stack = vec![e];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if integral_parts(n, pool).is_some() {
            map.insert(
                n,
                pool.symbol(format!("__opaque_int_{}", n.0), Domain::Real),
            );
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if map.is_empty() {
        return e;
    }
    super::separate::replace_nodes(e, &map, pool)
}

/// Combine `Σ cᵢ ∫ fᵢ dτ` over equal ranges (`cᵢ` free of `τ`) into one
/// integral, drop empty ranges, and replace an integral whose integrand is
/// proved zero by `0`.
pub(crate) fn merge_integrals(e: ExprId, pool: &ExprPool) -> ExprId {
    let canon = pool.symbol("__merge_tau", Domain::Real);
    let ex = super::separate::expanded(e, pool);
    let terms = match pool.get(ex) {
        ExprData::Add(xs) => xs,
        _ => vec![ex],
    };
    let mut groups: Vec<((ExprId, ExprId), Vec<ExprId>)> = Vec::new();
    let mut rest = Vec::new();
    'terms: for t in terms {
        let factors = match pool.get(t) {
            ExprData::Mul(xs) => xs,
            _ => vec![t],
        };
        let mut integral = None;
        let mut coef = Vec::new();
        for f in factors {
            if let Some(parts) = integral_parts(f, pool) {
                if integral.is_some() {
                    rest.push(t);
                    continue 'terms;
                }
                integral = Some(parts);
            } else {
                coef.push(f);
            }
        }
        let Some((b, tau, lo, hi)) = integral else {
            rest.push(t);
            continue;
        };
        if simp(sub(hi, lo, pool), pool) == pool.integer(0) {
            continue;
        }
        let c = pool.mul(coef);
        if mentions_var(c, tau, pool) {
            rest.push(t);
            continue;
        }
        let body = pool.mul(vec![c, subs1(b, tau, canon, pool)]);
        let key = (simp(lo, pool), simp(hi, pool));
        match groups.iter_mut().find(|(k, _)| *k == key) {
            Some((_, v)) => v.push(body),
            None => groups.push((key, vec![body])),
        }
    }
    for ((lo, hi), bodies) in groups {
        let b = simp(pool.add(bodies), pool);
        if proved(b, pool) {
            continue;
        }
        rest.push(pool.func(INTEGRAL, vec![b, canon, lo, hi]));
    }
    simp(pool.add(rest), pool)
}

/// Split `∫_{lo}^{hi}` at the breakpoints of `Piecewise` nodes in the
/// integration variable and integrate each smooth piece (the branch is
/// chosen at the piece's midpoint, with the parameters at sample positive
/// values, and only when two different samples agree).
pub(crate) fn definite_piecewise(
    body: ExprId,
    var: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut pw = Vec::new();
    let mut stack = vec![body];
    let mut seen = std::collections::HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Piecewise { .. } = pool.get(n) {
            pw.push(n);
            continue;
        }
        push_children(n, pool, &mut stack);
    }
    if pw.is_empty() {
        return None;
    }
    // Breakpoints from simple comparisons with the variable.
    let mut cuts: Vec<ExprId> = Vec::new();
    let mut preds = Vec::new();
    for &p in &pw {
        if let ExprData::Piecewise { branches, .. } = pool.get(p) {
            for (c, _) in branches {
                preds.push(c);
            }
        }
    }
    while let Some(c) = preds.pop() {
        if let ExprData::Predicate { kind, args } = pool.get(c) {
            use crate::kernel::expr::PredicateKind as K;
            match kind {
                K::And | K::Or | K::Not => preds.extend(args),
                K::Lt | K::Le | K::Gt | K::Ge | K::Eq | K::Ne if args.len() == 2 => {
                    let (a, b) = (args[0], args[1]);
                    let cut = if a == var && !mentions_var(b, var, pool) {
                        b
                    } else if b == var && !mentions_var(a, var, pool) {
                        a
                    } else {
                        return None;
                    };
                    cuts.push(simp(cut, pool));
                }
                K::True | K::False => {}
                _ => return None,
            }
        } else {
            return None;
        }
    }
    let params: Vec<ExprId> = symbols_of(body, pool)
        .into_iter()
        .chain(symbols_of(lo, pool))
        .chain(symbols_of(hi, pool))
        .filter(|&s| s != var && !is_constant_symbol(s, pool))
        .collect();
    let sample = |k: usize| -> HashMap<ExprId, f64> {
        let vals = [[1.3, 0.7, 1.9, 1.1, 2.3], [2.9, 1.6, 0.8, 3.7, 1.2]];
        params
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, vals[k][i % 5]))
            .collect()
    };
    let ev =
        |e: ExprId, env: &HashMap<ExprId, f64>| crate::jit::eval_interp_real(e, env, pool).ok();
    // Order the points by both samples; they must agree.
    let mut pts: Vec<ExprId> = vec![lo, hi];
    pts.extend(cuts);
    let mut orders = Vec::new();
    for k in 0..2 {
        let env = sample(k);
        let mut vals: Vec<(f64, ExprId)> = Vec::new();
        for &p in &pts {
            vals.push((ev(p, &env)?, p));
        }
        let (a, b) = (ev(lo, &env)?, ev(hi, &env)?);
        vals.retain(|&(v, _)| v >= a - 1e-12 && v <= b + 1e-12);
        vals.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        vals.dedup_by(|x, y| (x.0 - y.0).abs() < 1e-12);
        orders.push(vals.iter().map(|v| v.1).collect::<Vec<_>>());
    }
    if orders[0] != orders[1] {
        return None;
    }
    let pts = &orders[0];
    let mut total = Vec::new();
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let mid = pool.mul(vec![pool.rational(1, 2), pool.add(vec![a, b])]);
        // Choose each Piecewise branch at the midpoint (both samples agree).
        let mut map = HashMap::new();
        for &p in &pw {
            let ExprData::Piecewise { branches, default } = pool.get(p) else {
                continue;
            };
            let mut chosen: Option<ExprId> = None;
            for k in 0..2 {
                let mut env = sample(k);
                env.insert(var, ev(mid, &sample(k))?);
                let mut pick = default;
                for &(c, v) in &branches {
                    if eval_pred(c, &env, pool)? {
                        pick = v;
                        break;
                    }
                }
                match chosen {
                    None => chosen = Some(pick),
                    Some(q) if q == pick => {}
                    _ => return None,
                }
            }
            map.insert(p, chosen?);
        }
        let piece = simp(super::separate::replace_nodes(body, &map, pool), pool);
        total.push(definite(piece, var, a, b, pool)?);
    }
    Some(simp(pool.add(total), pool))
}

fn eval_pred(c: ExprId, env: &HashMap<ExprId, f64>, pool: &ExprPool) -> Option<bool> {
    use crate::kernel::expr::PredicateKind as K;
    let ExprData::Predicate { kind, args } = pool.get(c) else {
        return None;
    };
    let ev = |e: ExprId| crate::jit::eval_interp_real(e, env, pool).ok();
    Some(match kind {
        K::True => true,
        K::False => false,
        K::And => {
            let mut all = true;
            for a in args {
                all &= eval_pred(a, env, pool)?;
            }
            all
        }
        K::Or => {
            let mut any = false;
            for a in args {
                any |= eval_pred(a, env, pool)?;
            }
            any
        }
        K::Not => !eval_pred(args[0], env, pool)?,
        K::Lt => ev(args[0])? < ev(args[1])?,
        K::Le => ev(args[0])? <= ev(args[1])?,
        K::Gt => ev(args[0])? > ev(args[1])?,
        K::Ge => ev(args[0])? >= ev(args[1])?,
        K::Eq => (ev(args[0])? - ev(args[1])?).abs() < 1e-14,
        K::Ne => (ev(args[0])? - ev(args[1])?).abs() >= 1e-14,
    })
}
