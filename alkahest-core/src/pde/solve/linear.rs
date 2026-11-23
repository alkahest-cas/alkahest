//! Linear, semilinear and quasilinear first-order equations
//! `Σ aᵢ u_{xᵢ} = c` by the method of characteristics.

use std::collections::HashMap;

use crate::kernel::{ExprId, ExprPool};
use crate::ode::dsolve::{dsolve_with, OdeInput, SolutionForm};
use crate::simplify::assumptions::AssumptionContext;

use super::cauchy::GeneralForm;
use super::integrals::{IntegralSearch, VectorField};
use super::util::*;
use super::{tidy, Candidate, Ctx, PdeSolutionKind};

/// Indices of the variables, simplest coefficient first (a number before a
/// symbol before an expression).
fn order_by_simplicity(a: &[ExprId], pool: &ExprPool) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..a.len()).filter(|&i| !is_zero(a[i], pool)).collect();
    idx.sort_by_key(|&i| (if is_number(a[i], pool) { 0 } else { 1 }, pool.depth(a[i])));
    idx
}

/// `zⱼ = Zⱼ(z_k, K)` for every `j ≠ k` from `ξₘ(z) = Kₘ`, by sequential
/// isolation.
fn invert_integrals(
    integrals: &[ExprId],
    ks: &[ExprId],
    vars: &[ExprId],
    k: usize,
    pool: &ExprPool,
) -> Option<HashMap<ExprId, ExprId>> {
    let mut eqs: Vec<ExprId> = integrals
        .iter()
        .zip(ks)
        .map(|(&xi, &kk)| sub(xi, kk, pool))
        .collect();
    let mut unsolved: Vec<ExprId> = vars
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != k)
        .map(|(_, &v)| v)
        .collect();
    let mut sol: HashMap<ExprId, ExprId> = HashMap::new();
    while !unsolved.is_empty() {
        let mut progress = false;
        for ei in 0..eqs.len() {
            let e = eqs[ei];
            let present: Vec<ExprId> = unsolved
                .iter()
                .copied()
                .filter(|&v| contains(e, v, pool))
                .collect();
            if present.len() != 1 {
                continue;
            }
            let v = present[0];
            let Some(val) = isolate_one(e, v, pool) else {
                continue;
            };
            for e2 in eqs.iter_mut() {
                *e2 = simp(subs1(*e2, v, val, pool), pool);
            }
            for s in sol.values_mut() {
                *s = simp(subs1(*s, v, val, pool), pool);
            }
            sol.insert(v, val);
            unsolved.retain(|&x| x != v);
            eqs.remove(ei);
            progress = true;
            break;
        }
        if !progress {
            return None;
        }
    }
    Some(sol)
}

/// Linear and semilinear equations: the coefficient ratios `aᵢ/a_k` do not
/// involve `u`.
pub(crate) fn solve_linear(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let Some(aff) = ctx.affine.clone() else {
        return Vec::new();
    };
    let n = ctx.indep.len();
    let order = order_by_simplicity(&aff.a, pool);
    let Some(&k0) = order.first() else {
        return Vec::new();
    };
    // Normalise by a_k0 and require u-free coefficients.
    let a: Vec<ExprId> = aff
        .a
        .iter()
        .map(|&ai| simp(div(ai, aff.a[k0], pool), pool))
        .collect();
    if a.iter().any(|&ai| contains(ai, ctx.w, pool)) {
        ctx.fail("first_order_linear: the characteristic directions depend on u".into());
        return Vec::new();
    }
    let c = simp(div(aff.c, aff.a[k0], pool), pool);
    let field = VectorField::new(ctx.indep.clone(), a.clone(), pool);
    let mut search = IntegralSearch::new(&field, ctx.taken.clone());
    search.run(n - 1, pool);
    let integrals = search.found.clone();
    let mut steps = vec![format!(
        "characteristic system: {}",
        ctx.indep
            .iter()
            .zip(&a)
            .map(|(&x, &ai)| format!("d{}/({})", pool.display(x), pool.display(ai)))
            .chain(std::iter::once(format!("du/({})", pool.display(c))))
            .collect::<Vec<_>>()
            .join(" = ")
    )];
    steps.extend(search.notes.clone());
    if integrals.len() < n - 1 {
        ctx.fail(format!(
            "first_order_linear: found {} of {} independent first integrals",
            integrals.len(),
            n - 1
        ));
        return Vec::new();
    }
    let mut out = Vec::new();
    // Homogeneous: u is constant along characteristics, u = F1(ξ₁, …).  No
    // inversion of the integrals is needed (they may be unevaluated
    // integrals that cannot be inverted).
    if is_zero(c, pool) {
        let fname = ctx.arb.fresh_name();
        let fapp = pool.func(fname.clone(), integrals.clone());
        let mut st = steps.clone();
        st.push("homogeneous: u is constant along characteristics".into());
        out.push(Candidate {
            value: fapp,
            implicit: false,
            kind: PdeSolutionKind::General,
            constants: Vec::new(),
            side_conditions: Vec::new(),
            steps: st,
            general: Some(GeneralForm::Explicit {
                integrals: integrals.clone(),
                func: fname,
                value: fapp,
            }),
        });
        return out;
    }
    for &k in &order {
        let ks: Vec<ExprId> = (0..n - 1).map(|_| ctx.fresh_internal(pool)).collect();
        let Some(inv) = invert_integrals(&integrals, &ks, &ctx.indep, k, pool) else {
            continue;
        };
        let zk = ctx.indep[k];
        // du/dz_k = c / a_k along a characteristic.
        let rhs = simp(subs_map(div(c, a[k], pool), &inv, pool), pool);
        if ctx
            .indep
            .iter()
            .enumerate()
            .any(|(j, &zj)| j != k && contains(rhs, zj, pool))
        {
            continue;
        }
        let back: HashMap<ExprId, ExprId> =
            ks.iter().copied().zip(integrals.iter().copied()).collect();
        let fname = ctx.arb.fresh_name();
        let fapp = pool.func(fname.clone(), integrals.clone());
        // Quadrature when the reduced ODE does not involve u.
        let linear_in_u = d(rhs, ctx.w, pool).filter(|&al| !contains(al, ctx.w, pool));
        let branches: Vec<(SolutionForm, ExprId)> = if let Some(al) = linear_in_u {
            // u' = a u + b: u = exp(A) (C + integral of b exp(-A)), A the
            // antiderivative of a, with the quadratures done here (by parts
            // where the integrator alone declines).
            let beta = simp(subs1(rhs, ctx.w, pool.integer(0), pool), pool);
            let Some(ia) = quad(al, zk, pool) else {
                continue;
            };
            let mu = tidy(simp(pool.func("exp", vec![neg(ia, pool)]), pool), pool);
            let Some(ib) = quad(pool.mul(vec![beta, mu]), zk, pool) else {
                ctx.fail(format!(
                    "first_order_linear: the quadrature of {} in {} did not close",
                    pool.display(pool.mul(vec![beta, mu])),
                    pool.display(zk)
                ));
                continue;
            };
            let c1 = ctx.fresh_internal(pool);
            let y = pool.mul(vec![pool.func("exp", vec![ia]), pool.add(vec![ib, c1])]);
            vec![(SolutionForm::Explicit(y), c1)]
        } else {
            let (input, wp) = OdeInput::first_order(zk, ctx.w, pool);
            let input = input.with_equation(sub(wp, rhs, pool));
            match dsolve_with(&input, &AssumptionContext::new(), pool) {
                Ok(rep) => rep
                    .branches
                    .into_iter()
                    .filter_map(|b| b.constants.first().map(|&c1| (b.form, c1)))
                    .collect(),
                Err(e) => {
                    ctx.fail(format!("first_order_linear: reduced ODE: {e}"));
                    continue;
                }
            }
        };
        for (form, c1) in branches {
            match form {
                SolutionForm::Explicit(y) => {
                    let mut m = back.clone();
                    m.insert(c1, fapp);
                    let value = simp(subs_map(y, &m, pool), pool);
                    let mut st = steps.clone();
                    st.push(format!(
                        "along characteristics du/d{} = {}",
                        pool.display(zk),
                        pool.display(rhs)
                    ));
                    st.push(format!("integration constant → {}", pool.display(fapp)));
                    out.push(Candidate {
                        value,
                        implicit: false,
                        kind: PdeSolutionKind::General,
                        constants: Vec::new(),
                        side_conditions: Vec::new(),
                        steps: st,
                        general: Some(GeneralForm::Explicit {
                            integrals: integrals.clone(),
                            func: fname.clone(),
                            value,
                        }),
                    });
                }
                SolutionForm::Implicit(g) => {
                    // ξₙ(z, u) = C1 from the relation.
                    let Some(cv) = isolate_one(g, c1, pool) else {
                        continue;
                    };
                    let xi_n = simp(subs_map(cv, &back, pool), pool);
                    let mut all = integrals.clone();
                    all.push(xi_n);
                    let value = ctx.with_unknown(sub(xi_n, fapp, pool), pool);
                    let mut st = steps.clone();
                    st.push(format!(
                        "along characteristics: {} = const",
                        pool.display(xi_n)
                    ));
                    out.push(Candidate {
                        value,
                        implicit: true,
                        kind: PdeSolutionKind::Implicit,
                        constants: Vec::new(),
                        side_conditions: Vec::new(),
                        steps: st,
                        general: Some(GeneralForm::Integrals {
                            integrals: all,
                            affine: Some(aff.clone()),
                        }),
                    });
                }
            }
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

/// Quasilinear equations by Lagrange's method: `n` independent first
/// integrals of `dxᵢ/aᵢ = du/c`.
pub(crate) fn solve_quasilinear(ctx: &mut Ctx, pool: &ExprPool) -> Vec<Candidate> {
    let Some(aff) = ctx.affine.clone() else {
        return Vec::new();
    };
    let n = ctx.indep.len();
    let order = order_by_simplicity(&aff.a, pool);
    let Some(&k0) = order.first() else {
        return Vec::new();
    };
    let mut vars = ctx.indep.clone();
    vars.push(ctx.w);
    let mut comps: Vec<ExprId> = aff
        .a
        .iter()
        .map(|&ai| simp(div(ai, aff.a[k0], pool), pool))
        .collect();
    comps.push(simp(div(aff.c, aff.a[k0], pool), pool));
    let field = VectorField::new(vars, comps, pool);
    let mut search = IntegralSearch::new(&field, ctx.taken.clone());
    search.run(n, pool);
    let integrals = search.found.clone();
    let mut steps = vec!["Lagrange's method: first integrals of dxᵢ/aᵢ = du/c".to_string()];
    steps.extend(search.notes.clone());
    if integrals.len() < n {
        ctx.fail(format!(
            "first_order_quasilinear: found {} of {} independent first integrals",
            integrals.len(),
            n
        ));
        return Vec::new();
    }
    let (free, with_u): (Vec<ExprId>, Vec<ExprId>) = integrals
        .iter()
        .partition(|&&xi| !contains(xi, ctx.w, pool));
    let mut out = Vec::new();
    let general = GeneralForm::Integrals {
        integrals: integrals.clone(),
        affine: Some(aff.clone()),
    };
    if with_u.len() == 1 {
        let fname = ctx.arb.fresh_name();
        let fapp = pool.func(fname.clone(), free.clone());
        let rel = sub(with_u[0], fapp, pool);
        // Explicit when u isolates.
        for cand in isolate(rel, ctx.w, pool) {
            let mut st = steps.clone();
            st.push(format!(
                "{} = {} solved for u",
                pool.display(with_u[0]),
                pool.display(fapp)
            ));
            out.push(Candidate {
                value: cand,
                implicit: false,
                kind: PdeSolutionKind::General,
                constants: Vec::new(),
                side_conditions: Vec::new(),
                steps: st,
                general: Some(GeneralForm::Explicit {
                    integrals: free.clone(),
                    func: fname.clone(),
                    value: cand,
                }),
            });
        }
        let mut st = steps.clone();
        st.push("implicit general solution".into());
        out.push(Candidate {
            value: ctx.with_unknown(rel, pool),
            implicit: true,
            kind: PdeSolutionKind::Implicit,
            constants: Vec::new(),
            side_conditions: Vec::new(),
            steps: st,
            general: Some(general),
        });
    } else {
        let fapp = ctx.arb.fresh(integrals.clone(), pool);
        let mut st = steps.clone();
        st.push(format!(
            "implicit general solution {} = 0",
            pool.display(fapp)
        ));
        out.push(Candidate {
            value: ctx.with_unknown(fapp, pool),
            implicit: true,
            kind: PdeSolutionKind::Implicit,
            constants: Vec::new(),
            side_conditions: Vec::new(),
            steps: st,
            general: Some(general),
        });
    }
    out
}
