//! Riccati characteristics `y′ = q₀(x) + q₁(x) y + q₂(x) y²` with symbolic
//! parameters.
//!
//! The handbook equations of this type are built around a particular
//! solution of the form `c₁·h(x) + c₀`, where `h` is a power `x^e` or an
//! exponential `e^{λx}` and the exponent is fixed by balancing the terms of
//! `q₀`, `q₁` and `q₂`. For example, `y′ = a xⁿ y² + b m x^{m−1} − a b² x^{n+2m}` has
//! `y_p = b x^m`, and `y′ = y² + b y − a² e^{2λx} + a(λ − b) e^{λx}` has
//! `y_p = a e^{λx}`.
//!
//! The procedure:
//!
//! 1. For each shape, the residual `y_p′ − q₀ − q₁ y_p − q₂ y_p²` is split
//!    into its coefficients on the `x`-dependent generators.
//! 2. The resulting polynomial equations in `c₁, c₀` are solved over
//!    `ℚ(parameters)`.
//! 3. Each candidate is kept only when the residual is *proved* zero.
//! 4. `y = y_p + 1/v` then gives a linear ODE for `v`. That yields the first
//!    integral `v·e^{∫P} + ∫ q₂ e^{∫P} dx` with `P = q₁ + 2 q₂ y_p`. Unevaluated
//!    integrals are allowed there.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::kernel::{ExprData, ExprId, ExprPool};

use super::util::*;

/// `(q₀, q₁, q₂)` of a quadratic in `y`; `None` otherwise or when `q₂ = 0`.
fn quadratic(ratio: ExprId, y: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId, ExprId)> {
    let r = crate::simplify::engine::simplify_expanded(ratio, pool).value;
    let d1 = d(r, y, pool)?;
    let d2 = d(d1, y, pool)?;
    if contains(d2, y, pool) || is_zero(d2, pool) {
        return None;
    }
    let q2 = simp(pool.mul(vec![pool.rational(1, 2), d2]), pool);
    let zero = pool.integer(0);
    let q1 = simp(subs1(d1, y, zero, pool), pool);
    let q0 = simp(subs1(r, y, zero, pool), pool);
    Some((q0, q1, q2))
}

/// The factors of a term.
fn factors(t: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(t) {
        ExprData::Mul(xs) => xs,
        _ => vec![t],
    }
}

fn terms(q: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(q) {
        ExprData::Add(xs) => xs,
        _ => vec![q],
    }
}

/// Exponents `e` of `x` in the terms of `q` (structurally: `x`, `x^e`).
fn power_exponents(q: ExprId, x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    'terms: for t in terms(q, pool) {
        let mut es = Vec::new();
        for f in factors(t, pool) {
            if f == x {
                es.push(pool.integer(1));
            } else if let ExprData::Pow { base, exp } = pool.get(f) {
                if base == x && !contains(exp, x, pool) {
                    es.push(exp);
                } else if contains(f, x, pool) {
                    continue 'terms;
                }
            } else if contains(f, x, pool) {
                continue 'terms;
            }
        }
        out.push(simp(pool.add(es), pool));
    }
    out
}

/// Rates `λ` of `e^{λx}` in the terms of `q` (structurally: `exp(λx + c)`).
fn exp_rates(q: ExprId, x: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    'terms: for t in terms(q, pool) {
        let mut ls = Vec::new();
        for f in factors(t, pool) {
            let (base, k) = match pool.get(f) {
                ExprData::Pow { base, exp } if !contains(exp, x, pool) => (base, Some(exp)),
                _ => (f, None),
            };
            match pool.get(base) {
                ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                    let Some(l) = d(args[0], x, pool) else {
                        continue 'terms;
                    };
                    if contains(l, x, pool) {
                        continue 'terms;
                    }
                    ls.push(match k {
                        Some(k) => pool.mul(vec![k, l]),
                        None => l,
                    });
                }
                _ if contains(f, x, pool) => continue 'terms,
                _ => {}
            }
        }
        out.push(simp(pool.add(ls), pool));
    }
    out
}

/// Each term of the expanded `e` with its powers of `x` gathered into one
/// `x^{Σe}` and its exponentials into one `exp(Σ)`, so equal monomials are
/// one generator for [`coefficient_equations`].
fn normalize_monomials(e: ExprId, x: ExprId, pool: &ExprPool) -> ExprId {
    let e = crate::simplify::engine::simplify_expanded(e, pool).value;
    let mut out = Vec::new();
    for t in terms(e, pool) {
        let mut pw = Vec::new();
        let mut ex = Vec::new();
        let mut rest = Vec::new();
        for f in factors(t, pool) {
            let (base, k) = match pool.get(f) {
                ExprData::Pow { base, exp } if !contains(exp, x, pool) => (base, exp),
                _ => (f, pool.integer(1)),
            };
            if base == x {
                pw.push(k);
                continue;
            }
            match pool.get(base) {
                ExprData::Pow { base: b2, exp: e2 } if b2 == x && !contains(e2, x, pool) => {
                    pw.push(pool.mul(vec![k, e2]));
                }
                ExprData::Func { name, args } if name == "exp" && args.len() == 1 => {
                    ex.push(pool.mul(vec![k, args[0]]));
                }
                _ => rest.push(f),
            }
        }
        if !pw.is_empty() {
            let s = simp(pool.add(pw), pool);
            rest.push(pool.pow(x, s));
        }
        if !ex.is_empty() {
            let s = crate::simplify::engine::simplify_expanded(pool.add(ex), pool).value;
            rest.push(pool.func("exp", vec![s]));
        }
        out.push(pool.mul(rest));
    }
    pool.add(out)
}

/// Polynomial equations in `unknowns` from `r ≡ 0` identically in `x`:
/// the coefficients of `r`'s numerator on the `x`-dependent generators.
fn coefficient_equations(
    r: ExprId,
    unknowns: &[ExprId],
    x: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    let r = normalize_monomials(r, x, pool);
    let mut vars = unknowns.to_vec();
    vars.push(x);
    let (rf, gens) = crate::poly::cancel::rational_over_generators(r, vars, pool).ok()?;
    let xdep: Vec<bool> = gens.iter().map(|&g| contains(g, x, pool)).collect();
    let mut groups: BTreeMap<Vec<u32>, Vec<ExprId>> = BTreeMap::new();
    for (exps, coeff) in &rf.numer.terms {
        let get = |i: usize| exps.get(i).copied().unwrap_or(0);
        let key: Vec<u32> = (0..gens.len()).filter(|&i| xdep[i]).map(get).collect();
        let mut factors = vec![pool.integer(coeff.clone())];
        for (i, &g) in gens.iter().enumerate() {
            let e = get(i);
            if !xdep[i] && e > 0 {
                factors.push(pool.pow(g, pool.integer(e)));
            }
        }
        groups.entry(key).or_default().push(pool.mul(factors));
    }
    Some(
        groups
            .into_values()
            .map(|t| simp(pool.add(t), pool))
            .collect(),
    )
}

/// Solve the equations for the unknowns (sequential isolation); every
/// solution is returned as a substitution map.
fn solve_equations(
    eqs: Vec<ExprId>,
    unknowns: &[ExprId],
    pool: &ExprPool,
    depth: usize,
) -> Vec<std::collections::HashMap<ExprId, ExprId>> {
    let eqs: Vec<ExprId> = eqs.into_iter().filter(|&e| !is_zero(e, pool)).collect();
    let pending: Vec<ExprId> = unknowns
        .iter()
        .copied()
        .filter(|&u| eqs.iter().any(|&e| contains(e, u, pool)))
        .collect();
    if eqs.is_empty() {
        return vec![std::collections::HashMap::new()];
    }
    if pending.is_empty() || depth > 4 {
        return Vec::new();
    }
    // An equation with one pending unknown, simplest first.
    let mut order: Vec<ExprId> = eqs.clone();
    order.sort_by_key(|&e| pool.depth(e));
    for &e in &order {
        let present: Vec<ExprId> = pending
            .iter()
            .copied()
            .filter(|&u| contains(e, u, pool))
            .collect();
        if present.len() != 1 {
            continue;
        }
        let u = present[0];
        let mut out = Vec::new();
        for val in isolate(e, u, pool).into_iter().take(3) {
            if unknowns.iter().any(|&w| contains(val, w, pool)) {
                continue;
            }
            let rest: Vec<ExprId> = eqs
                .iter()
                .filter(|&&x| x != e)
                .map(|&x| simp(subs1(x, u, val, pool), pool))
                .collect();
            let others: Vec<ExprId> = unknowns.iter().copied().filter(|&w| w != u).collect();
            for mut m in solve_equations(rest, &others, pool, depth + 1) {
                let v = simp(subs_map(val, &m, pool), pool);
                m.insert(u, v);
                out.push(m);
            }
        }
        return out;
    }
    // Every equation still holds both unknowns: fix the constant shift to 0.
    if let Some(&c0) = pending.last() {
        let zero = pool.integer(0);
        let eqs2: Vec<ExprId> = eqs
            .iter()
            .map(|&e| simp(subs1(e, c0, zero, pool), pool))
            .collect();
        let others: Vec<ExprId> = unknowns.iter().copied().filter(|&w| w != c0).collect();
        return solve_equations(eqs2, &others, pool, depth + 1)
            .into_iter()
            .map(|mut m| {
                m.insert(c0, zero);
                m
            })
            .collect();
    }
    Vec::new()
}

/// Particular solutions of `y′ = q₀ + q₁ y + q₂ y²` of the shape
/// `c₁ h + c₀`.
fn particular_solutions(
    q: (ExprId, ExprId, ExprId),
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Vec<ExprId> {
    let (q0, q1, q2) = q;
    let rhs = pool.add(vec![
        q0,
        pool.mul(vec![q1, y]),
        pool.mul(vec![q2, pool.pow(y, pool.integer(2))]),
    ]);
    let half = pool.rational(1, 2);
    let mut shapes: BTreeSet<ExprId> = BTreeSet::new();
    let (e0, e1, e2) = (
        power_exponents(q0, x, pool),
        power_exponents(q1, x, pool),
        power_exponents(q2, x, pool),
    );
    let push_pow = |e: ExprId, set: &mut BTreeSet<ExprId>| {
        let e = simp(e, pool);
        if !contains(e, x, pool) {
            set.insert(simp(pool.pow(x, e), pool));
        }
    };
    push_pow(pool.integer(-1), &mut shapes);
    for &a in &e0 {
        for &b in &e2 {
            push_pow(pool.mul(vec![half, sub(a, b, pool)]), &mut shapes);
        }
        push_pow(pool.add(vec![a, pool.integer(1)]), &mut shapes);
        for &c in &e1 {
            push_pow(sub(a, c, pool), &mut shapes);
        }
    }
    for &c in &e1 {
        for &b in &e2 {
            push_pow(sub(c, b, pool), &mut shapes);
        }
    }
    let (l0, l1, l2) = (
        exp_rates(q0, x, pool),
        exp_rates(q1, x, pool),
        exp_rates(q2, x, pool),
    );
    let push_exp = |l: ExprId, set: &mut BTreeSet<ExprId>| {
        let l = simp(l, pool);
        if !contains(l, x, pool) && !is_zero(l, pool) {
            set.insert(simp(pool.func("exp", vec![pool.mul(vec![l, x])]), pool));
        }
    };
    for &a in &l0 {
        for &b in &l2 {
            push_exp(pool.mul(vec![half, sub(a, b, pool)]), &mut shapes);
        }
        for &c in &l1 {
            push_exp(sub(a, c, pool), &mut shapes);
        }
    }
    for &c in &l1 {
        for &b in &l2 {
            push_exp(sub(c, b, pool), &mut shapes);
        }
    }
    let mut taken = names_in(&[rhs], pool);
    let c1 = fresh_symbol("__pde_r", &mut taken, pool);
    let c0 = fresh_symbol("__pde_r", &mut taken, pool);
    let mut out = Vec::new();
    // The constant shape first.
    let mut cands: Vec<ExprId> = vec![c0];
    for h in shapes.iter().take(24) {
        cands.push(pool.add(vec![pool.mul(vec![c1, *h]), c0]));
    }
    for yp in cands {
        let Some(dyp) = d(yp, x, pool) else { continue };
        let res = sub(dyp, subs1(rhs, y, yp, pool), pool);
        let res = crate::simplify::engine::simplify_expanded(res, pool).value;
        let Some(eqs) = coefficient_equations(res, &[c1, c0], x, pool) else {
            continue;
        };
        for m in solve_equations(eqs, &[c1, c0], pool, 0) {
            let mut m = m;
            m.entry(c1).or_insert(pool.integer(0));
            m.entry(c0).or_insert(pool.integer(0));
            let cand = simp(subs_map(yp, &m, pool), pool);
            if contains(cand, c1, pool) || contains(cand, c0, pool) || contains(cand, y, pool) {
                continue;
            }
            let Some(dc) = d(cand, x, pool) else { continue };
            if is_zero(sub(dc, subs1(rhs, y, cand, pool), pool), pool) && !out.contains(&cand) {
                out.push(cand);
            }
        }
        if out.len() >= 2 {
            break;
        }
    }
    out
}

/// The classical linearisation of a Riccati equation: with
/// `y = −w′/(q₂ w)`, `y′ = q₀ + q₁ y + q₂ y²` becomes the linear second-order
/// equation
///
/// ```text
/// w″ − (q₁ + q₂′/q₂) w′ + q₀ q₂ w = 0,
/// ```
///
/// which `dsolve` may solve in Airy, Bessel, Whittaker or elementary
/// functions.  From `w = C₁w₁ + C₂w₂` the ratio `−C₁/C₂` is constant along
/// solutions, so
///
/// ```text
/// ξ = (w₂′ + q₂ y w₂) / (w₁′ + q₂ y w₁)
/// ```
///
/// is a first integral.  Candidates only: the caller checks `V ξ ≡ 0`.
fn linear_reduction_integrals(
    q: (ExprId, ExprId, ExprId),
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Vec<ExprId> {
    use crate::ode::dsolve::{dsolve_with, OdeInput, SolutionForm};
    let (q0, q1, q2) = q;
    let Some(dq2) = d(q2, x, pool) else {
        return Vec::new();
    };
    let mut taken = names_in(&[q0, q1, q2, x, y], pool);
    let w = fresh_symbol("__pde_w", &mut taken, pool);
    let (input, wp, wpp) = OdeInput::second_order(x, w, pool);
    let a1 = simp(pool.add(vec![q1, div(dq2, q2, pool)]), pool);
    let eq = pool.add(vec![
        wpp,
        neg(pool.mul(vec![a1, wp]), pool),
        pool.mul(vec![q0, q2, w]),
    ]);
    let input = input.with_equation(simp(eq, pool));
    let Ok(rep) = dsolve_with(
        &input,
        &crate::simplify::assumptions::AssumptionContext::new(),
        pool,
    ) else {
        return Vec::new();
    };
    let zero = pool.integer(0);
    let mut out = Vec::new();
    for b in rep.branches {
        let SolutionForm::Explicit(wx) = b.form else {
            continue;
        };
        if b.constants.len() != 2 || contains(wx, y, pool) {
            continue;
        }
        let (c1, c2) = (b.constants[0], b.constants[1]);
        // w = C₁ w₁ + C₂ w₂ exactly.
        let (Some(w1), Some(w2)) = (d(wx, c1, pool), d(wx, c2, pool)) else {
            continue;
        };
        if contains(w1, c1, pool)
            || contains(w1, c2, pool)
            || contains(w2, c1, pool)
            || contains(w2, c2, pool)
            || has_undefined(w1, pool)
            || has_undefined(w2, pool)
        {
            continue;
        }
        let mut m = HashMap::new();
        m.insert(c1, zero);
        m.insert(c2, zero);
        if !is_zero(subs_map(wx, &m, pool), pool) {
            continue;
        }
        let (Some(w1p), Some(w2p)) = (d(w1, x, pool), d(w2, x, pool)) else {
            continue;
        };
        let num = pool.add(vec![w2p, pool.mul(vec![q2, y, w2])]);
        let den = pool.add(vec![w1p, pool.mul(vec![q2, y, w1])]);
        out.push(simp(div(num, den, pool), pool));
        if out.len() >= 2 {
            break;
        }
    }
    out
}

/// First-integral candidates of `dy/dx = ratio` when it is a Riccati
/// equation with a particular solution of the shapes above, or (when none is
/// found) through its linearisation ([`linear_reduction_integrals`]).
pub(crate) fn riccati_integrals(
    ratio: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Vec<ExprId> {
    let Some(q) = quadratic(ratio, y, pool) else {
        return Vec::new();
    };
    let (_, q1, q2) = q;
    let mut out = Vec::new();
    let yps = particular_solutions(q, x, y, pool);
    if yps.is_empty() {
        out.extend(linear_reduction_integrals(q, x, y, pool));
    }
    for yp in yps {
        let p = simp(
            pool.add(vec![q1, pool.mul(vec![pool.integer(2), q2, yp])]),
            pool,
        );
        for unev in [false, true] {
            let ip = if unev {
                unevaluated(p, x, pool)
            } else {
                quad(p, x, pool)
            };
            let Some(ip) = ip else { continue };
            let e = simp(pool.func("exp", vec![ip]), pool);
            let integrand = pool.mul(vec![q2, e]);
            let iq = if unev {
                unevaluated(integrand, x, pool)
            } else {
                quad(integrand, x, pool)
            };
            let Some(iq) = iq else { continue };
            let v = recip(sub(y, yp, pool), pool);
            out.push(simp(pool.add(vec![pool.mul(vec![v, e]), iq]), pool));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn show(pool: &ExprPool, es: &[ExprId]) -> Vec<String> {
        es.iter().map(|&e| pool.display(e).to_string()).collect()
    }

    /// Local debugging aid: `cargo test … debug_riccati -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn debug_riccati() {
        let pool = ExprPool::new();
        let mut syms = std::collections::HashMap::new();
        syms.insert("x".to_string(), pool.symbol("x", Domain::Real));
        syms.insert("y".to_string(), pool.symbol("y", Domain::Real));
        for src in [
            "-a*b^2*x^(2*m + n) + a*x^n*y^2 + b*m*x^(m - 1)",
            "-a^2*exp(2*lamda*x) + a*(-b + lamda)*exp(lamda*x) + b*y + y^2",
            "y^2 - 2/x^2",
        ] {
            let r = crate::parse::parse(src, &pool, &mut syms).unwrap();
            let x = pool.symbol("x", Domain::Real);
            let y = pool.symbol("y", Domain::Real);
            let q = quadratic(r, y, &pool).unwrap();
            eprintln!("q = {:?}", show(&pool, &[q.0, q.1, q.2]));
            eprintln!("e0 = {:?}", show(&pool, &power_exponents(q.0, x, &pool)));
            eprintln!(
                "yp = {:?}",
                show(&pool, &particular_solutions(q, x, y, &pool))
            );
            let mut taken = names_in(&[r], &pool);
            let c1 = fresh_symbol("__pde_r", &mut taken, &pool);
            let c0 = fresh_symbol("__pde_r", &mut taken, &pool);
            let h = crate::parse::parse(
                if src.contains("lamda") {
                    "exp(lamda*x)"
                } else if src.contains("m") {
                    "x^m"
                } else {
                    "x^(-1)"
                },
                &pool,
                &mut syms,
            )
            .unwrap();
            let yp = pool.add(vec![pool.mul(vec![c1, h]), c0]);
            let res = sub(d(yp, x, &pool).unwrap(), subs1(r, y, yp, &pool), &pool);
            let res = crate::simplify::engine::simplify_expanded(res, &pool).value;
            eprintln!("res = {}", pool.display(res));
            if let Some(eqs) = coefficient_equations(res, &[c1, c0], x, &pool) {
                eprintln!("eqs = {:?}", show(&pool, &eqs));
                eprintln!("sols = {}", solve_equations(eqs, &[c1, c0], &pool, 0).len());
            }
        }
    }

    #[test]
    fn handbook_riccati_particular_solutions() {
        let pool = ExprPool::new();
        let mut syms = std::collections::HashMap::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        syms.insert("x".to_string(), x);
        syms.insert("y".to_string(), y);
        // y' = y² − 2/x² has y_p = −1/x (and 2/x).
        let r = crate::parse::parse("y^2 - 2/x^2", &pool, &mut syms).unwrap();
        let q = quadratic(r, y, &pool).unwrap();
        assert!(!particular_solutions(q, x, y, &pool).is_empty());
        assert!(!riccati_integrals(r, x, y, &pool).is_empty());
    }
}
