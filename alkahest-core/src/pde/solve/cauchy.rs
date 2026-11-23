//! Cauchy problems: a general solution specialised to data `u = φ(s)` on a
//! non-characteristic manifold `x = x(s)` (typically `u(x, 0) = f(x)`).
//!
//! * The data manifold is read off the condition `u(a₁(s), …, aₙ(s)) = φ(s)`:
//!   its parameters are the independent variables occurring in the
//!   arguments (`u(x, 0)` → one parameter `x`), and there must be `n − 1`.
//! * Non-characteristic check: the Jacobian of the data manifold against the
//!   characteristic direction must not vanish identically (`E-PDE-052` when
//!   it does); when it is not a constant it is recorded as a side condition.
//! * Elimination: the first integrals restricted to the data are solved for
//!   the parameters (sequential isolation, with the polynomial /
//!   transcendental solver behind it) and substituted back.
//! * For `u_t + c(u) u_x = 0` the **breaking time** `t_b = t₀ − 1/min g′`,
//!   `g = c(φ)`, is reported as the side condition `t_b − t > 0`.

use std::collections::HashMap;

use crate::deriv::log::SideCondition;
use crate::diff::undefined::{replace_function, view};
use crate::kernel::{ExprId, ExprPool};

use super::super::error::PdeError;
use super::util::*;
use super::{Affine, Candidate, Ctx, PdeSolutionKind};

/// The structure of a general solution, kept by a tier for Cauchy fitting.
#[derive(Clone, Debug)]
pub(crate) enum GeneralForm {
    /// `u = value`, where `value` contains `func(ξ₁, …, ξₙ₋₁)` and the `ξ`
    /// are free of `u`.
    Explicit {
        integrals: Vec<ExprId>,
        func: String,
        value: ExprId,
    },
    /// `n` first integrals `ξ(x, u)` of the characteristic system; the
    /// general solution is `Φ(ξ₁, …, ξₙ) = 0`.
    Integrals {
        integrals: Vec<ExprId>,
        affine: Option<Affine>,
    },
}

struct Data {
    /// Curve parameters (fresh symbols) and the original variables they
    /// replace.
    params: Vec<ExprId>,
    originals: Vec<ExprId>,
    /// `aᵢ(s)`.
    args: Vec<ExprId>,
    /// `φ(s)`.
    phi: ExprId,
}

fn read_data(ctx: &mut Ctx, pool: &ExprPool) -> Result<Data, PdeError> {
    let conds = ctx.problem.conditions();
    if conds.len() != 1 {
        return Err(PdeError::CauchyUnsupported(format!(
            "a first-order Cauchy problem takes one data condition; got {}",
            conds.len()
        )));
    }
    let c = &conds[0];
    let v = view(c.lhs, pool).ok_or_else(|| {
        PdeError::CauchyUnsupported(format!("{} is not u evaluated on a curve", pool.display(c.lhs)))
    })?;
    if v.inner != c.lhs {
        return Err(PdeError::CauchyUnsupported(format!(
            "{}: data on a derivative are not Cauchy data for a first-order equation",
            pool.display(c.lhs)
        )));
    }
    let n = ctx.indep.len();
    let originals: Vec<ExprId> = ctx
        .indep
        .iter()
        .copied()
        .filter(|&x| v.args.iter().any(|&a| contains(a, x, pool)))
        .collect();
    if originals.len() != n - 1 {
        return Err(PdeError::CauchyUnsupported(format!(
            "the data {} = {} lie on a manifold with {} parameters; {} are needed",
            pool.display(c.lhs),
            pool.display(c.rhs),
            originals.len(),
            n - 1
        )));
    }
    let params: Vec<ExprId> = originals.iter().map(|_| ctx.fresh_internal(pool)).collect();
    let m: HashMap<ExprId, ExprId> = originals.iter().copied().zip(params.iter().copied()).collect();
    let args = v.args.iter().map(|&a| simp(subs_map(a, &m, pool), pool)).collect();
    let phi = simp(subs_map(c.rhs, &m, pool), pool);
    if !free_of(phi, &ctx.indep, pool) {
        return Err(PdeError::CauchyUnsupported(format!(
            "the data value {} depends on a variable that is not a curve parameter",
            pool.display(c.rhs)
        )));
    }
    Ok(Data {
        params,
        originals,
        args,
        phi,
    })
}

/// Restrict `e(z, w)` to the data: `z → a(s)`, `w → φ(s)`.
fn on_data(ctx: &Ctx, e: ExprId, data: &Data, pool: &ExprPool) -> ExprId {
    let mut m: HashMap<ExprId, ExprId> =
        ctx.indep.iter().copied().zip(data.args.iter().copied()).collect();
    m.insert(ctx.w, data.phi);
    simp(subs_map(e, &m, pool), pool)
}

fn det(m: &[Vec<ExprId>], pool: &ExprPool) -> ExprId {
    match m.len() {
        0 => pool.integer(1),
        1 => m[0][0],
        _ => {
            let mut terms = Vec::new();
            for j in 0..m.len() {
                let minor: Vec<Vec<ExprId>> = m[1..]
                    .iter()
                    .map(|r| {
                        r.iter()
                            .enumerate()
                            .filter(|&(c, _)| c != j)
                            .map(|(_, &e)| e)
                            .collect()
                    })
                    .collect();
                let sgn = if j % 2 == 0 { 1 } else { -1 };
                terms.push(pool.mul(vec![pool.integer(sgn), m[0][j], det(&minor, pool)]));
            }
            simp(pool.add(terms), pool)
        }
    }
}

/// Solve `σₘ(s) = Σₘ` for the parameters `s`.
fn solve_params(
    sig: &[ExprId],
    targets: &[ExprId],
    params: &[ExprId],
    pool: &ExprPool,
) -> Option<HashMap<ExprId, ExprId>> {
    let mut eqs: Vec<ExprId> = sig.iter().zip(targets).map(|(&a, &b)| sub(a, b, pool)).collect();
    let mut unsolved = params.to_vec();
    let mut sol: HashMap<ExprId, ExprId> = HashMap::new();
    while !unsolved.is_empty() {
        let mut progress = false;
        for ei in 0..eqs.len() {
            let present: Vec<ExprId> = unsolved
                .iter()
                .copied()
                .filter(|&v| contains(eqs[ei], v, pool))
                .collect();
            if present.len() != 1 {
                continue;
            }
            let v = present[0];
            let Some(val) = isolate_one(eqs[ei], v, pool) else {
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

fn back_to_originals(e: ExprId, data: &Data, pool: &ExprPool) -> ExprId {
    let m: HashMap<ExprId, ExprId> = data
        .params
        .iter()
        .copied()
        .zip(data.originals.iter().copied())
        .collect();
    simp(subs_map(e, &m, pool), pool)
}

/// Specialise a general-solution candidate to the problem's data.
pub(crate) fn impose(
    ctx: &mut Ctx,
    cand: Candidate,
    pool: &ExprPool,
) -> Result<Vec<Candidate>, PdeError> {
    let Some(general) = cand.general.clone() else {
        return Err(PdeError::CauchyUnsupported(format!(
            "a {} answer cannot be fitted to data",
            cand.kind.as_str()
        )));
    };
    let data = read_data(ctx, pool)?;
    let mut steps = cand.steps.clone();
    match general {
        GeneralForm::Explicit {
            integrals,
            func,
            value,
        } => {
            let sig: Vec<ExprId> = integrals.iter().map(|&xi| on_data(ctx, xi, &data, pool)).collect();
            let jac: Vec<Vec<ExprId>> = sig
                .iter()
                .map(|&s| {
                    data.params
                        .iter()
                        .map(|&p| d(s, p, pool).unwrap_or(s))
                        .collect()
                })
                .collect();
            let j = det(&jac, pool);
            if is_zero(j, pool) {
                return Err(PdeError::CharacteristicData(format!(
                    "the integrals {} are constant on the data manifold",
                    integrals
                        .iter()
                        .map(|&x| pool.display(x).to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            let mut side = cand.side_conditions.clone();
            if !is_number(j, pool) {
                side.push(SideCondition::NonZero(back_to_originals(j, &data, pool)));
            }
            let sigmas: Vec<ExprId> = integrals.iter().map(|_| ctx.fresh_internal(pool)).collect();
            let Some(s_of) = solve_params(&sig, &sigmas, &data.params, pool) else {
                return Err(PdeError::CauchyUnsupported(
                    "could not solve the data manifold for its parameters".into(),
                ));
            };
            // u on the data, with the arbitrary function replaced by a symbol.
            let phi_sym = ctx.fresh_internal(pool);
            let pattern = pool.func(func.clone(), sigmas.clone());
            let val_phi = replace_function(value, pattern, phi_sym, pool)
                .map_err(|e| PdeError::CauchyUnsupported(e.to_string()))?;
            let eq = sub(on_data(ctx, val_phi, &data, pool), data.phi, pool);
            let mut out = Vec::new();
            for f_of_s in isolate(eq, phi_sym, pool) {
                let body = simp(subs_map(f_of_s, &s_of, pool), pool);
                let Ok(fitted) = replace_function(value, pattern, body, pool) else {
                    continue;
                };
                let mut st = steps.clone();
                st.push(format!(
                    "data imposed: {}({}) = {}",
                    func,
                    sigmas
                        .iter()
                        .map(|&x| pool.display(x).to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    pool.display(body)
                ));
                out.push(Candidate {
                    value: simp(fitted, pool),
                    implicit: false,
                    kind: PdeSolutionKind::Particular,
                    constants: Vec::new(),
                    side_conditions: side.clone(),
                    steps: st,
                    general: None,
                });
            }
            if out.is_empty() {
                return Err(PdeError::CauchyUnsupported(
                    "the arbitrary function could not be solved from the data".into(),
                ));
            }
            Ok(out)
        }
        GeneralForm::Integrals { integrals, affine } => {
            let n = ctx.indep.len();
            let sig: Vec<ExprId> = integrals.iter().map(|&xi| on_data(ctx, xi, &data, pool)).collect();
            let mut side = cand.side_conditions.clone();
            // Non-characteristic: det[ ∂a/∂s ; characteristic direction ].
            if let Some(aff) = &affine {
                let mut rows: Vec<Vec<ExprId>> = data
                    .params
                    .iter()
                    .map(|&p| {
                        data.args
                            .iter()
                            .map(|&a| d(a, p, pool).unwrap_or(a))
                            .collect()
                    })
                    .collect();
                rows.push(aff.a.iter().map(|&ai| on_data(ctx, ai, &data, pool)).collect());
                let j = det(&rows, pool);
                if is_zero(j, pool) {
                    return Err(PdeError::CharacteristicData(
                        "the data curve is tangent to the characteristic direction everywhere"
                            .into(),
                    ));
                }
                if !is_number(j, pool) {
                    side.push(SideCondition::NonZero(back_to_originals(j, &data, pool)));
                }
                if let Some((tb, note)) = breaking_time(ctx, aff, &data, pool) {
                    steps.push(note);
                    if let Some((t, tbv)) = tb {
                        side.push(SideCondition::Positive(simp(sub(tbv, t, pool), pool)));
                    }
                }
            }
            // Choose n − 1 integrals to solve for the parameters; the rest
            // gives the relation.
            let mut out = Vec::new();
            for rest in (0..n).rev() {
                let chosen: Vec<usize> = (0..n).filter(|&i| i != rest).collect();
                let csig: Vec<ExprId> = chosen.iter().map(|&i| sig[i]).collect();
                let targets: Vec<ExprId> = chosen.iter().map(|&i| integrals[i]).collect();
                let Some(s_of) = solve_params(&csig, &targets, &data.params, pool) else {
                    continue;
                };
                let rel = simp(
                    sub(integrals[rest], subs_map(sig[rest], &s_of, pool), pool),
                    pool,
                );
                if !contains(rel, ctx.w, pool) {
                    continue;
                }
                for e in isolate(rel, ctx.w, pool) {
                    let mut st = steps.clone();
                    st.push(format!("data imposed; solved for u: u = {}", pool.display(e)));
                    out.push(Candidate {
                        value: e,
                        implicit: false,
                        kind: PdeSolutionKind::Particular,
                        constants: Vec::new(),
                        side_conditions: side.clone(),
                        steps: st,
                        general: None,
                    });
                }
                let mut st = steps.clone();
                st.push(format!(
                    "data imposed; implicit solution {} = 0",
                    pool.display(ctx.with_unknown(rel, pool))
                ));
                out.push(Candidate {
                    value: ctx.with_unknown(rel, pool),
                    implicit: true,
                    kind: PdeSolutionKind::Implicit,
                    constants: Vec::new(),
                    side_conditions: side.clone(),
                    steps: st,
                    general: None,
                });
                break;
            }
            if out.is_empty() {
                return Err(PdeError::CauchyUnsupported(
                    "the data parameters could not be eliminated from the first integrals".into(),
                ));
            }
            Ok(out)
        }
    }
}

type Breaking = (Option<(ExprId, ExprId)>, String);

/// `u_t + c(u) u_x = 0` with data at `t = t₀`: the breaking time
/// `t_b = t₀ − 1/min g′(s)`, `g = c(φ(s))`.
fn breaking_time(ctx: &Ctx, aff: &Affine, data: &Data, pool: &ExprPool) -> Option<Breaking> {
    if ctx.indep.len() != 2 || !is_zero(aff.c, pool) {
        return None;
    }
    // The "time" slot is the one held constant by the data.
    let tk = (0..2).find(|&k| free_of(data.args[k], &data.params, pool))?;
    let xk = 1 - tk;
    if data.args[xk] != data.params[0] {
        return None;
    }
    let speed = simp(div(aff.a[xk], aff.a[tk], pool), pool);
    if !free_of(speed, &ctx.indep, pool) || !contains(speed, ctx.w, pool) {
        return None;
    }
    let s = data.params[0];
    let t = ctx.indep[tk];
    let t0 = data.args[tk];
    let g = simp(subs1(speed, ctx.w, data.phi, pool), pool);
    let gp = d(g, s, pool)?;
    let x = data.originals[0];
    if is_number(gp, pool) {
        let r = as_rational(gp, pool)?;
        if r >= 0 {
            return Some((
                None,
                format!(
                    "characteristics do not cross for {} > {} (g' = {} ≥ 0): no breaking",
                    pool.display(t),
                    pool.display(t0),
                    pool.display(gp)
                ),
            ));
        }
        let tb = simp(sub(t0, recip(gp, pool), pool), pool);
        return Some((
            Some((t, tb)),
            format!("breaking time t_b = {} (g' = {})", pool.display(tb), pool.display(gp)),
        ));
    }
    // Non-constant g': the global minimum, numerically located and
    // confirmed at an exact critical point.
    let mut best = f64::INFINITY;
    for i in 0..=4000 {
        let sv = -20.0 + 40.0 * (i as f64) / 4000.0;
        let mut env = HashMap::new();
        env.insert(s, sv);
        if let Some(v) = eval_at(gp, &env, pool) {
            best = best.min(v);
        }
    }
    let gpp = d(gp, s, pool)?;
    if !best.is_finite() {
        return None;
    }
    if best >= 0.0 {
        return Some((
            None,
            format!(
                "g' = {} appears non-negative: no breaking for {} > {}",
                pool.display(subs1(gp, s, x, pool)),
                pool.display(t),
                pool.display(t0)
            ),
        ));
    }
    let mut taken = ctx.taken.clone();
    let zero = pool.integer(0);
    for crit in isolate(gpp, s, pool) {
        // A periodic family's index symbol is set to 0.
        let syms: Vec<ExprId> = symbols(crit, pool).into_iter().collect();
        let mut m = HashMap::new();
        for sy in syms {
            m.insert(sy, zero);
        }
        let crit0 = simp(subs_map(crit, &m, pool), pool);
        let val = simp(subs1(gp, s, crit0, pool), pool);
        let Some(v) = eval_at(val, &HashMap::new(), pool) else {
            continue;
        };
        if (v - best).abs() <= 1e-6 * (1.0 + best.abs()) {
            let tb = simp(sub(t0, recip(val, pool), pool), pool);
            let _ = &mut taken;
            return Some((
                Some((t, tb)),
                format!(
                    "breaking time t_b = {} (min g' = {} at {} = {})",
                    pool.display(tb),
                    pool.display(val),
                    pool.display(x),
                    pool.display(crit0)
                ),
            ));
        }
    }
    Some((
        None,
        format!(
            "characteristics cross (min g' ≈ {best:.6} < 0, t_b ≈ {:.6}); exact minimum not found",
            -1.0 / best
        ),
    ))
}
