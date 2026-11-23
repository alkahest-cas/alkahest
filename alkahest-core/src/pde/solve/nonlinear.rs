//! Fully nonlinear first-order equations `F(x, u, p) = 0`: complete
//! integrals by the standard forms and by Lagrange–Charpit, singular
//! integrals by envelope elimination.
//!
//! Every complete integral keeps its constants symbolic (`C1, C2, …`) and is
//! verified as an identity in them.

use std::collections::HashMap;

use crate::kernel::{ExprData, ExprId, ExprPool};

use super::integrals::{IntegralSearch, VectorField};
use super::util::*;
use super::{Candidate, Ctx, PdeSolutionKind};

fn complete(value: ExprId, implicit: bool, constants: Vec<ExprId>, steps: Vec<String>) -> Candidate {
    Candidate {
        value,
        implicit,
        kind: PdeSolutionKind::CompleteIntegral,
        constants,
        side_conditions: Vec::new(),
        steps,
        general: None,
    }
}

/// `u = Σ xᵢpᵢ + f(p)` ⇒ `u = Σ aᵢxᵢ + f(a)`.
pub(crate) fn solve_clairaut(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    if !contains(ctx.f, ctx.w, pool) {
        return Vec::new();
    }
    let omega = ctx.fresh_internal(pool);
    let lin = pool.add(
        ctx.indep
            .iter()
            .zip(&ctx.p)
            .map(|(&x, &p)| pool.mul(vec![x, p]))
            .chain(std::iter::once(omega))
            .collect(),
    );
    let g = simp(subs1(ctx.f, ctx.w, lin, pool), pool);
    if !free_of(g, &ctx.indep, pool) || !contains(g, omega, pool) {
        return Vec::new();
    }
    let consts: Vec<ExprId> = (0..ctx.p.len()).map(|_| ctx.fresh_constant(pool)).collect();
    let mut out = Vec::new();
    for f in isolate(g, omega, pool) {
        let m: HashMap<ExprId, ExprId> = ctx.p.iter().copied().zip(consts.iter().copied()).collect();
        let fa = subs_map(f, &m, pool);
        let value = pool.add(
            ctx.indep
                .iter()
                .zip(&consts)
                .map(|(&x, &a)| pool.mul(vec![a, x]))
                .chain(std::iter::once(fa))
                .collect(),
        );
        out.push(complete(
            value,
            false,
            consts.clone(),
            vec![
                format!("Clairaut form u = Σ xᵢpᵢ + {}", pool.display(f)),
                "complete integral: pᵢ → constants".into(),
            ],
        ));
    }
    out
}

/// `f(p₁, …, pₙ) = 0` ⇒ `u = Σ_{i≠k} aᵢxᵢ + Q(a) x_k + b`.
pub(crate) fn solve_f_p(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let f = ctx.f;
    if contains(f, ctx.w, pool) || !free_of(f, &ctx.indep, pool) {
        return Vec::new();
    }
    let n = ctx.indep.len();
    let mut out = Vec::new();
    for k in (0..n).rev() {
        if !contains(f, ctx.p[k], pool) {
            continue;
        }
        let roots = isolate(f, ctx.p[k], pool);
        if roots.is_empty() {
            continue;
        }
        let consts: Vec<ExprId> = (0..n).map(|_| ctx.fresh_constant(pool)).collect();
        let mut m = HashMap::new();
        let mut ci = 0;
        for i in 0..n {
            if i != k {
                m.insert(ctx.p[i], consts[ci]);
                ci += 1;
            }
        }
        for q in roots {
            let qa = subs_map(q, &m, pool);
            let mut terms = Vec::new();
            let mut ci = 0;
            for i in 0..n {
                if i == k {
                    terms.push(pool.mul(vec![qa, ctx.indep[i]]));
                } else {
                    terms.push(pool.mul(vec![consts[ci], ctx.indep[i]]));
                    ci += 1;
                }
            }
            terms.push(consts[n - 1]);
            out.push(complete(
                pool.add(terms),
                false,
                consts.clone(),
                vec![format!(
                    "f(p) = 0: {} = {}",
                    pool.display(ctx.p[k]),
                    pool.display(q)
                )],
            ));
        }
        break;
    }
    out
}

/// `f(u, p, q) = 0` ⇒ `u = U(ζ)`, `ζ = x + a y`, `f(U, U′, aU′) = 0`.
pub(crate) fn solve_f_up(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let f = ctx.f;
    if !contains(f, ctx.w, pool) || !free_of(f, &ctx.indep, pool) {
        return Vec::new();
    }
    let (x, y) = (ctx.indep[0], ctx.indep[1]);
    let a = ctx.fresh_constant(pool);
    let b = ctx.fresh_constant(pool);
    let v = ctx.fresh_internal(pool);
    let mut m = HashMap::new();
    m.insert(ctx.p[0], v);
    m.insert(ctx.p[1], pool.mul(vec![a, v]));
    let h_eq = simp(subs_map(f, &m, pool), pool);
    let zeta = pool.add(vec![x, pool.mul(vec![a, y])]);
    let mut out = Vec::new();
    for h in isolate(h_eq, v, pool) {
        if is_zero(h, pool) {
            continue;
        }
        let Some(hint) = integrate(recip(h, pool), ctx.w, pool) else {
            continue;
        };
        let rel = simp(sub(hint, pool.add(vec![zeta, b]), pool), pool);
        let steps = vec![
            format!("u = U(ζ), ζ = {}", pool.display(zeta)),
            format!("U′ = {}", pool.display(h)),
            format!("∫ dU/U′ = {}", pool.display(hint)),
        ];
        for e in isolate(rel, ctx.w, pool) {
            out.push(complete(e, false, vec![a, b], steps.clone()));
        }
        out.push(complete(ctx.with_unknown(rel, pool), true, vec![a, b], steps));
    }
    out
}

/// Terms of a sum.
fn terms_of(e: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    match pool.get(e) {
        ExprData::Add(xs) => xs,
        _ => vec![e],
    }
}

/// `Σ fᵢ(xᵢ, pᵢ) + c₀ = 0` ⇒ `fᵢ = aᵢ`, `u = Σ ∫ Pᵢ dxᵢ + b`.
pub(crate) fn solve_separable(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let f = ctx.f;
    if contains(f, ctx.w, pool) {
        return Vec::new();
    }
    let n = ctx.indep.len();
    let mut groups: Vec<Vec<ExprId>> = vec![Vec::new(); n];
    let mut c0 = Vec::new();
    for t in terms_of(f, pool) {
        let idx: Vec<usize> = (0..n)
            .filter(|&i| contains(t, ctx.indep[i], pool) || contains(t, ctx.p[i], pool))
            .collect();
        match idx.len() {
            0 => c0.push(t),
            1 => groups[idx[0]].push(t),
            _ => return Vec::new(),
        }
    }
    if groups.iter().any(|g| g.is_empty()) {
        return Vec::new();
    }
    if groups
        .iter()
        .enumerate()
        .any(|(i, g)| !contains(pool.add(g.clone()), ctx.p[i], pool))
    {
        return Vec::new();
    }
    let consts: Vec<ExprId> = (0..n).map(|_| ctx.fresh_constant(pool)).collect();
    let mut sum_a = Vec::new();
    let mut terms = Vec::new();
    let mut steps = vec!["additively separable: Σ fᵢ(xᵢ, pᵢ) = const".to_string()];
    for i in 0..n {
        let fi = pool.add(groups[i].clone());
        let rhs = if i + 1 < n {
            sum_a.push(consts[i]);
            consts[i]
        } else {
            neg(pool.add(c0.iter().copied().chain(sum_a.iter().copied()).collect()), pool)
        };
        let Some(pi) = isolate_one(sub(fi, rhs, pool), ctx.p[i], pool) else {
            return Vec::new();
        };
        let Some(ii) = integrate(pi, ctx.indep[i], pool) else {
            return Vec::new();
        };
        steps.push(format!("{} = {}", pool.display(ctx.p[i]), pool.display(pi)));
        terms.push(ii);
    }
    terms.push(consts[n - 1]);
    vec![complete(pool.add(terms), false, consts, steps)]
}

/// Lagrange–Charpit for `F(x, y, u, p, q) = 0`.
pub(crate) fn solve_charpit(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let f = ctx.f;
    let (x, y) = (ctx.indep[0], ctx.indep[1]);
    let (p, q, w) = (ctx.p[0], ctx.p[1], ctx.w);
    let (Some(fp), Some(fq), Some(fx), Some(fy), Some(fu)) =
        (d(f, p, pool), d(f, q, pool), d(f, x, pool), d(f, y, pool), d(f, w, pool))
    else {
        return Vec::new();
    };
    let comps = vec![
        fp,
        fq,
        pool.add(vec![pool.mul(vec![p, fp]), pool.mul(vec![q, fq])]),
        neg(pool.add(vec![fx, pool.mul(vec![p, fu])]), pool),
        neg(pool.add(vec![fy, pool.mul(vec![q, fu])]), pool),
    ];
    let field = VectorField::new(vec![x, y, w, p, q], comps, pool);
    let mut search = IntegralSearch::new(&field, ctx.taken.clone());
    if !search.offer(f, "the equation itself", pool) {
        return Vec::new();
    }
    search.run(2, pool);
    if search.found.len() < 2 {
        ctx.fail("charpit: no second first integral of Charpit's system".into());
        return Vec::new();
    }
    let phi = search.found[1];
    let a = ctx.fresh_constant(pool);
    let b = ctx.fresh_constant(pool);
    let mut steps = vec!["Charpit's system".to_string()];
    steps.extend(search.notes.clone());
    // Solve {F = 0, φ = a} for p, q.
    let eqs = [f, sub(phi, a, pool)];
    let mut out = Vec::new();
    for order in [[p, q], [q, p]] {
        for (e1, e2) in [(0, 1), (1, 0)] {
            // e1 gives order[0] in terms of order[1]; e2 then gives order[1].
            for v0 in isolate(eqs[e1], order[0], pool) {
                let e2s = simp(subs1(eqs[e2], order[0], v0, pool), pool);
                if contains(e2s, order[0], pool) {
                    continue;
                }
                for v1 in isolate(e2s, order[1], pool) {
                    let v0s = simp(subs1(v0, order[1], v1, pool), pool);
                    let (pp, qq) = if order[0] == p { (v0s, v1) } else { (v1, v0s) };
                    if !free_of(pp, &[p, q], pool) || !free_of(qq, &[p, q], pool) {
                        continue;
                    }
                    let mut st = steps.clone();
                    st.push(format!(
                        "p = {}, q = {}",
                        pool.display(pp),
                        pool.display(qq)
                    ));
                    out.extend(pfaff(ctx, pp, qq, a, b, st, pool));
                    if out.len() >= 4 {
                        return out;
                    }
                }
            }
        }
    }
    out
}

/// Integrate `du = P dx + Q dy`.
fn pfaff(
    ctx: &Ctx,
    pp: ExprId,
    qq: ExprId,
    a: ExprId,
    b: ExprId,
    steps: Vec<String>,
    pool: &ExprPool,
) -> Vec<Candidate> {
    let (x, y, w) = (ctx.indep[0], ctx.indep[1], ctx.w);
    let mut out = Vec::new();
    let (scale, pt, qt) = if !contains(pp, w, pool) && !contains(qq, w, pool) {
        (None, pp, qq)
    } else {
        // P = m(u) P̃(x, y), Q = m(u) Q̃(x, y).
        let probe = if is_zero(pp, pool) { qq } else { pp };
        let Some(g) = d(pool.func("log", vec![probe]), w, pool) else {
            return out;
        };
        if !free_of(g, &[x, y], pool) {
            return out;
        }
        let Some(lg) = integrate(g, w, pool) else {
            return out;
        };
        let m = simp(pool.func("exp", vec![lg]), pool);
        let pt = simp(div(pp, m, pool), pool);
        let qt = simp(div(qq, m, pool), pool);
        if contains(pt, w, pool) || contains(qt, w, pool) {
            return out;
        }
        (Some(m), pt, qt)
    };
    let Some(i1) = integrate(pt, x, pool) else {
        return out;
    };
    let Some(di1) = d(i1, y, pool) else {
        return out;
    };
    let r = simp(sub(qt, di1, pool), pool);
    if contains(r, x, pool) {
        return out;
    }
    let Some(i2) = integrate(r, y, pool) else {
        return out;
    };
    let rhs = pool.add(vec![i1, i2, b]);
    match scale {
        None => out.push(complete(rhs, false, vec![a, b], steps)),
        Some(m) => {
            let Some(h) = integrate(recip(m, pool), w, pool) else {
                return out;
            };
            let rel = simp(sub(h, rhs, pool), pool);
            for e in isolate(rel, w, pool) {
                out.push(complete(e, false, vec![a, b], steps.clone()));
            }
            out.push(complete(ctx.with_unknown(rel, pool), true, vec![a, b], steps));
        }
    }
    out
}

/// The envelope of a complete integral `u = φ(x, a)`: `∂φ/∂aᵢ = 0` solved
/// for the constants and substituted.
pub(crate) fn singular_integral(
    ctx: &Ctx,
    value: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Option<Candidate> {
    if constants.is_empty() {
        return None;
    }
    let mut eqs = Vec::new();
    for &c in constants {
        let dc = d(value, c, pool)?;
        if is_number(dc, pool) && !is_zero(dc, pool) {
            return None;
        }
        eqs.push(dc);
    }
    let mut sol: HashMap<ExprId, ExprId> = HashMap::new();
    let mut unsolved = constants.to_vec();
    while !unsolved.is_empty() {
        let mut progress = false;
        for ei in 0..eqs.len() {
            let present: Vec<ExprId> = unsolved
                .iter()
                .copied()
                .filter(|&v| contains(eqs[ei], v, pool))
                .collect();
            if present.is_empty() {
                continue;
            }
            for &v in &present {
                let Some(val) = isolate_one(eqs[ei], v, pool) else {
                    continue;
                };
                if !free_of(val, &unsolved, pool) && present.len() == 1 {
                    continue;
                }
                for e2 in eqs.iter_mut() {
                    *e2 = simp(subs1(*e2, v, val, pool), pool);
                }
                for s in sol.values_mut() {
                    *s = simp(subs1(*s, v, val, pool), pool);
                }
                sol.insert(v, val);
                unsolved.retain(|&z| z != v);
                eqs.remove(ei);
                progress = true;
                break;
            }
            if progress {
                break;
            }
        }
        if !progress {
            return None;
        }
    }
    let s = simp(subs_map(value, &sol, pool), pool);
    if !free_of(s, constants, pool) || contains(s, ctx.w, pool) {
        return None;
    }
    Some(Candidate {
        value: s,
        implicit: false,
        kind: PdeSolutionKind::SingularIntegral,
        constants: Vec::new(),
        side_conditions: Vec::new(),
        steps: vec!["envelope: ∂u/∂(constants) = 0 eliminated".into()],
        general: None,
    })
}
