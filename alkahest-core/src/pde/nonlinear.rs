//! Nonlinear PDEs: reductions to ODEs and exact solution families
//! (milestone M6) — the equivalent of Maple PDEtools' `TWSolutions`,
//! `InvariantSolutions` and `SimilaritySolutions`.
//!
//! * [`reduce_by_generator`] — the invariants of a point-symmetry generator
//!   `X = ξ∂_x + τ∂_t + η∂_u` (characteristics by `dsolve`) give a similarity
//!   variable `s(x, t)` and the form `u = ψ(x, t, F(s))`; substituted into the
//!   equation they leave an ODE for `F`.
//! * [`traveling_wave_solutions`] — the generator `c∂_x + ∂_t`: `u = F(x − ct)`.
//!   The reduced ODE is integrated as far as it is an exact derivative
//!   (Euler operator), then solved by `dsolve` — **general solutions first**,
//!   as Kudryashov's "seven common errors" asks — and, when the equation is
//!   polynomial, by the tanh / sech / tan expansion methods with automatic
//!   balancing whose coefficient systems go to the F4 Gröbner solver.
//! * [`invariant_solutions`] / [`similarity_solutions`] — the same for any
//!   generator, and for combinations `X + λY` with `λ` fixed by requiring the
//!   reduced ODE to be exact (the Barenblatt and heat-source solutions).
//!
//! **Every** returned solution is verified by substitution into the original
//! equation through M0's [`verify_pde_solution`]; candidates that do not
//! verify exactly are discarded.

use std::collections::{HashMap, HashSet};

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

use super::error::PdeError;
use super::problem::PdeProblem;
use super::solve::util::{contains, d, fresh_symbol, is_number, is_zero, isolate, names_in, simp};
use super::symmetry::Jets;
use super::verify::{verify_pde_solution, PdeVerification, VerificationStatus};

/// Internal wall-clock caps (seconds) of the open-ended steps; an outer
/// budget that is tighter wins.
const ANSATZ_SECONDS: u64 = 12;
const REDUCE_SECONDS: u64 = 12;

/// Run `f` under a wall-clock cap of `secs` (or the remaining outer budget,
/// whichever is smaller); a trip of the cap alone is cleared afterwards so
/// it does not stop the caller.
pub(crate) fn capped<T>(secs: u64, f: impl FnOnce() -> T) -> T {
    use std::time::Duration;
    let cap = Duration::from_secs(secs);
    let mut b = crate::budget::Budget::new();
    match crate::budget::remaining() {
        Some((outer, _)) => {
            b.wall = Some(outer.wall.map_or(cap, |w| w.min(cap)));
            b.max_steps = outer.max_steps;
            b.seed = outer.seed;
        }
        None => b.wall = Some(cap),
    }
    let r = {
        let _g = crate::budget::enter(b);
        f()
    };
    if crate::budget::tripped().is_none() {
        crate::budget::clear_trip();
    }
    r
}

/// A verified solution family produced by a reduction.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ReducedSolution {
    /// `u = value`.
    pub value: ExprId,
    /// `"traveling_wave"`, `"tanh"`, `"sech"`, `"tan"`, `"invariant"`,
    /// `"similarity"`, `"cole_hopf"`.
    pub method: &'static str,
    /// Free constants of the family (`c`, `C1`, `k`, …).
    pub constants: Vec<ExprId>,
    /// Conditions the derivation imposed (`K1 = 0`: a decaying wave).
    pub assumptions: Vec<String>,
    /// Derivation steps.
    pub steps: Vec<String>,
    /// Always `exactly_verified`.
    pub verification: PdeVerification,
}

/// The reduction of an equation by a generator.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Reduction {
    /// The similarity variable `s(x, t)`.
    pub variable: ExprId,
    /// The symbol standing for `s` in the ODE.
    pub symbol: ExprId,
    /// The invariant form `u = ψ(x, t, F(s))`, written with the applied
    /// function `F(s(x, t))`.
    pub form: ExprId,
    /// The reduced ODE `= 0` in `F(symbol)` and its derivatives.
    pub ode: ExprId,
    /// The applied function `F(symbol)`.
    pub function: ExprId,
}

/// Internal reduction data (the ODE in symbols `Y_k = F^{(k)}(S)`).
#[derive(Clone, Debug)]
pub(crate) struct Red {
    pub s_expr: ExprId,
    pub s_sym: ExprId,
    pub form: ExprId,
    pub ys: Vec<ExprId>,
    pub ode: ExprId,
    pub order: usize,
    pub steps: Vec<String>,
}

fn nice_symbol(stem: &str, taken: &mut HashSet<String>, pool: &ExprPool) -> ExprId {
    if taken.insert(stem.to_string()) {
        return pool.symbol(stem, Domain::Real);
    }
    fresh_symbol(stem, taken, pool)
}

fn sub_m(e: ExprId, m: &HashMap<ExprId, ExprId>, pool: &ExprPool) -> ExprId {
    crate::kernel::subs::subs(e, m, pool)
}

fn minus(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])])
}

fn quotient(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, pool.pow(b, pool.integer(-1))])
}

/// Solve the first-order ODE `y' = rhs(z, y)` for `y(z)`; the solution and
/// its single constant.
fn solve_first_order(
    rhs: ExprId,
    y: ExprId,
    z: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let yp = pool.symbol("__red_yp", Domain::Real);
    let input = crate::ode::dsolve::OdeInput {
        x: z,
        y,
        derivs: vec![yp],
        equation: minus(yp, rhs, pool),
    };
    let res = crate::ode::dsolve::dsolve(&input, pool).ok()?;
    let sol = res.solutions.into_iter().find(|s| s.constants.len() == 1)?;
    Some((sol.y_of_x, sol.constants[0]))
}

/// The characteristic invariant `s(a, b)` of `ξ_a ∂_a + ξ_b ∂_b`.
fn similarity_variable(
    a: ExprId,
    b: ExprId,
    xa: ExprId,
    xb: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    if is_zero(xa, pool) {
        return Some(a);
    }
    if is_zero(xb, pool) {
        return Some(b);
    }
    let ratio = simp(quotient(xa, xb, pool), pool);
    // da/db = ratio(a, b).
    let s = if !contains(ratio, a, pool) {
        let i = super::solve::util::integrate(ratio, b, pool)?;
        simp(minus(a, i, pool), pool)
    } else {
        let y = pool.symbol("__red_y", Domain::Real);
        let r = crate::kernel::subs::subs(ratio, &HashMap::from([(a, y)]), pool);
        let (sol, c) = solve_first_order(r, y, b, pool)?;
        let eq = minus(a, sol, pool);
        let cands = isolate(eq, c, pool);
        let s = cands.into_iter().find(|&v| !contains(v, c, pool))?;
        simp(s, pool)
    };
    // X(s) = 0 exactly.
    let xs = pool.add(vec![
        pool.mul(vec![xa, d(s, a, pool)?]),
        pool.mul(vec![xb, d(s, b, pool)?]),
    ]);
    if !is_zero(xs, pool) {
        return None;
    }
    Some(s)
}

/// Reduce `f` (in jets) by the generator `comps = (ξ_a, ξ_b, η)` on two
/// independent variables.
pub(crate) fn reduce(
    jets: &Jets,
    f: ExprId,
    comps: &[ExprId],
    taken: &mut HashSet<String>,
    s_stem: &str,
    pool: &ExprPool,
) -> Result<Red, PdeError> {
    if jets.p() != 2 || comps.len() != 3 {
        return Err(PdeError::ReductionFailed(
            "reductions need two independent variables and a generator (ξ, τ, η)".into(),
        ));
    }
    let (a, b) = (jets.indep[0], jets.indep[1]);
    let w = jets.w;
    let (xa, xb, eta) = (
        simp(comps[0], pool),
        simp(comps[1], pool),
        simp(comps[2], pool),
    );
    if contains(xa, w, pool) || contains(xb, w, pool) {
        return Err(PdeError::ReductionFailed(
            "the generator moves the independent variables depending on u".into(),
        ));
    }
    if is_zero(xa, pool) && is_zero(xb, pool) {
        return Err(PdeError::ReductionFailed(
            "the generator does not move the independent variables".into(),
        ));
    }
    let s_expr = similarity_variable(a, b, xa, xb, pool).ok_or_else(|| {
        PdeError::ReductionFailed("the similarity variable could not be found".into())
    })?;
    let s_sym = nice_symbol(s_stem, taken, pool);
    let order = jets
        .jets_in(f, pool)
        .iter()
        .map(|(al, _)| al.iter().sum::<u32>())
        .max()
        .unwrap_or(0) as usize;
    let ys: Vec<ExprId> = (0..=2 * order + 3)
        .map(|k| fresh_symbol(&format!("__F{k}_"), taken, pool))
        .collect();
    // Parameter along the orbit and the other variable from s.
    let (param, other, xp) = if is_zero(xb, pool) {
        (a, b, xa)
    } else {
        (b, a, xb)
    };
    let other_of = if s_expr == other {
        None
    } else {
        let cands = isolate(minus(s_expr, s_sym, pool), other, pool);
        cands.into_iter().find(|&v| !contains(v, other, pool))
    };
    let form = if is_zero(eta, pool) {
        ys[0]
    } else {
        let y = pool.symbol("__red_u", Domain::Real);
        let mut m = HashMap::from([(w, y)]);
        if let Some(o) = other_of {
            m.insert(other, o);
        } else if s_expr != other {
            return Err(PdeError::ReductionFailed(
                "the orbit could not be parametrised".into(),
            ));
        } else {
            m.insert(other, s_sym);
        }
        let rhs = simp(sub_m(quotient(eta, xp, pool), &m, pool), pool);
        let (sol, c) = solve_first_order(rhs, y, param, pool).ok_or_else(|| {
            PdeError::ReductionFailed("the invariant of u could not be found".into())
        })?;
        let back = HashMap::from([(c, ys[0]), (s_sym, s_expr)]);
        simp(sub_m(sol, &back, pool), pool)
    };
    // u_J by the chain rule through s(a, b).
    let tot = |e: ExprId, i: usize| -> ExprId {
        let z = jets.indep[i];
        let ds = d(s_expr, z, pool).unwrap_or_else(|| pool.integer(0));
        let mut t = vec![d(e, z, pool).unwrap_or_else(|| pool.integer(0))];
        for k in 0..ys.len() - 1 {
            if let Some(p) = d(e, ys[k], pool) {
                if p != pool.integer(0) {
                    t.push(pool.mul(vec![ys[k + 1], ds, p]));
                }
            }
        }
        simp(pool.add(t), pool)
    };
    let mut memo: HashMap<Vec<u32>, ExprId> = HashMap::new();
    memo.insert(vec![0, 0], form);
    let mut m = HashMap::new();
    let mut js = jets.jets_in(f, pool);
    js.sort_by_key(|(al, _)| al.iter().sum::<u32>());
    for (al, sym) in js {
        let v = derive(&al, &mut memo, &tot);
        m.insert(sym, v);
    }
    let r = simp(sub_m(f, &m, pool), pool);
    // Eliminate the orbit parameter's partner: other = other(s, param).
    let r = if s_expr == other {
        sub_m(r, &HashMap::from([(other, s_sym)]), pool)
    } else {
        let o = other_of.ok_or_else(|| {
            PdeError::ReductionFailed("the similarity variable cannot be inverted".into())
        })?;
        sub_m(r, &HashMap::from([(other, o)]), pool)
    };
    let r = simp(r, pool);
    let top_k = (0..ys.len())
        .rev()
        .find(|&k| contains(r, ys[k], pool))
        .ok_or_else(|| {
            PdeError::ReductionFailed("the reduced equation has no unknown left".into())
        })?;
    let top = d(r, ys[top_k], pool)
        .ok_or_else(|| PdeError::ReductionFailed("cannot differentiate".into()))?;
    let ratio = crate::simplify::engine::simplify_expanded(quotient(r, top, pool), pool).value;
    let ode = if contains(ratio, param, pool) {
        let dp = d(ratio, param, pool).unwrap_or(ratio);
        if !is_zero(dp, pool) {
            return Err(PdeError::ReductionFailed(format!(
                "the reduced equation still depends on {}",
                pool.display(param)
            )));
        }
        simp(
            sub_m(ratio, &HashMap::from([(param, pool.integer(1))]), pool),
            pool,
        )
    } else {
        ratio
    };
    // Clear denominators.
    let ode = match crate::poly::cancel::together_parts(ode, ys[..=top_k].to_vec(), pool) {
        Ok((n, _)) if !is_zero(n, pool) => simp(n, pool),
        _ => ode,
    };
    Ok(Red {
        s_expr,
        s_sym,
        form,
        ys,
        ode,
        order: top_k,
        steps: vec![format!(
            "similarity variable {} = {}, u = {}",
            pool.display(s_sym),
            pool.display(s_expr),
            pool.display(form)
        )],
    })
}

fn derive(
    al: &[u32],
    memo: &mut HashMap<Vec<u32>, ExprId>,
    tot: &dyn Fn(ExprId, usize) -> ExprId,
) -> ExprId {
    if let Some(&v) = memo.get(al) {
        return v;
    }
    let i = al.iter().position(|&k| k > 0).expect("nonzero");
    let mut parent = al.to_vec();
    parent[i] -= 1;
    let base = derive(&parent, memo, tot);
    let v = tot(base, i);
    memo.insert(al.to_vec(), v);
    v
}

// ---------------------------------------------------------------------------
// One-dimensional Euler operator and exact-derivative inversion
// ---------------------------------------------------------------------------

fn d_s(e: ExprId, s: ExprId, ys: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut t = vec![d(e, s, pool).unwrap_or_else(|| pool.integer(0))];
    for k in 0..ys.len() - 1 {
        if let Some(p) = d(e, ys[k], pool) {
            if p != pool.integer(0) {
                t.push(pool.mul(vec![ys[k + 1], p]));
            }
        }
    }
    simp(pool.add(t), pool)
}

fn euler1(e: ExprId, s: ExprId, ys: &[ExprId], pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    for k in 0..ys.len() {
        if !contains(e, ys[k], pool) {
            continue;
        }
        let mut p = d(e, ys[k], pool).unwrap_or_else(|| pool.integer(0));
        for _ in 0..k {
            p = simp(pool.mul(vec![pool.integer(-1), d_s(p, s, ys, pool)]), pool);
        }
        terms.push(p);
    }
    if terms.is_empty() {
        return pool.integer(0);
    }
    crate::simplify::engine::simplify_expanded(pool.add(terms), pool).value
}

/// `P` with `D_s P = e` (`e` exact), or `None`.
fn invert1(e: ExprId, s: ExprId, ys: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut r = e;
    let mut parts = Vec::new();
    for _ in 0..24 {
        if is_zero(r, pool) {
            return Some(simp(pool.add(parts), pool));
        }
        let k = (0..ys.len()).rev().find(|&k| contains(r, ys[k], pool));
        match k {
            None => {
                let i = super::solve::util::integrate(r, s, pool)?;
                parts.push(i);
                r = simp(minus(r, d(i, s, pool)?, pool), pool);
            }
            Some(0) => return None,
            Some(k) => {
                let a = d(r, ys[k], pool)?;
                if contains(a, ys[k], pool) {
                    return None;
                }
                let p = super::solve::util::integrate(a, ys[k - 1], pool)?;
                if !is_zero(minus(d(p, ys[k - 1], pool)?, a, pool), pool) {
                    return None;
                }
                parts.push(p);
                r = crate::simplify::engine::simplify_expanded(
                    minus(r, d_s(p, s, ys, pool), pool),
                    pool,
                )
                .value;
            }
        }
    }
    None
}

/// Values of `lam` making `ode` an exact derivative.
fn exactness_values(
    ode: ExprId,
    lam: ExprId,
    s: ExprId,
    ys: &[ExprId],
    pool: &ExprPool,
) -> Vec<ExprId> {
    let e = euler1(ode, s, ys, pool);
    if is_zero(e, pool) {
        return vec![];
    }
    let mut gens = ys.to_vec();
    gens.push(s);
    let Ok((rf, g)) = crate::poly::cancel::rational_over_generators(
        super::detsys::canon_powers(e, pool),
        gens.clone(),
        pool,
    ) else {
        return vec![];
    };
    let lam_free: Vec<usize> = (0..g.len())
        .filter(|&i| !contains(g[i], lam, pool))
        .collect();
    let mut groups: HashMap<Vec<u32>, Vec<(Vec<u32>, rug::Integer)>> = HashMap::new();
    for (exps, c) in &rf.numer.terms {
        let get = |i: usize| exps.get(i).copied().unwrap_or(0);
        let key: Vec<u32> = lam_free.iter().map(|&i| get(i)).collect();
        let rest: Vec<u32> = (0..g.len())
            .map(|i| if lam_free.contains(&i) { 0 } else { get(i) })
            .collect();
        groups.entry(key).or_default().push((rest, c.clone()));
    }
    let lam_vars: Vec<ExprId> = (0..g.len())
        .map(|i| {
            if lam_free.contains(&i) {
                pool.integer(1)
            } else {
                g[i]
            }
        })
        .collect();
    let eqs: Vec<ExprId> = groups
        .into_values()
        .map(|ts| {
            let mut tm = std::collections::BTreeMap::new();
            for (mut k, c) in ts {
                while k.last() == Some(&0) {
                    k.pop();
                }
                tm.insert(k, c);
            }
            crate::poly::multipoly::MultiPoly {
                vars: lam_vars.clone(),
                terms: tm,
            }
            .to_expr(pool)
        })
        .collect();
    let Some(first) = eqs.iter().find(|&&q| contains(q, lam, pool)) else {
        return vec![];
    };
    let mut out = Vec::new();
    for v in isolate(*first, lam, pool) {
        if contains(v, lam, pool) {
            continue;
        }
        let m = HashMap::from([(lam, v)]);
        if eqs.iter().all(|&q| is_zero(sub_m(q, &m, pool), pool)) {
            let v = simp(v, pool);
            if !out.contains(&v) {
                out.push(v);
            }
        }
    }
    out
}

/// A solution `F(S)` of the reduced ODE with its constants and assumptions.
#[derive(Clone, Debug)]
struct OdeSol {
    f: ExprId,
    constants: Vec<ExprId>,
    assumptions: Vec<String>,
    steps: Vec<String>,
}

/// Integrate the exact part, then `dsolve`; with the integration constants
/// set to zero as a fallback.
fn solve_reduced(red: &Red, taken: &mut HashSet<String>, pool: &ExprPool) -> Vec<OdeSol> {
    let ys = &red.ys;
    let s = red.s_sym;
    let mut ode = red.ode;
    let mut ks: Vec<ExprId> = Vec::new();
    let mut steps = Vec::new();
    for _ in 0..3 {
        let order = (0..ys.len())
            .rev()
            .find(|&k| contains(ode, ys[k], pool))
            .unwrap_or(0);
        if order < 2 {
            break;
        }
        if !is_zero(euler1(ode, s, ys, pool), pool) {
            break;
        }
        let Some(p) = invert1(ode, s, ys, pool) else {
            break;
        };
        let k = nice_symbol(&format!("K{}", ks.len() + 1), taken, pool);
        steps.push(format!(
            "the reduced ODE is exact: {} = {}",
            pool.display(p),
            pool.display(k)
        ));
        ode = simp(minus(p, k, pool), pool);
        ks.push(k);
    }
    let mut out = Vec::new();
    let mut tries: Vec<(ExprId, Vec<ExprId>, Vec<String>)> = vec![(ode, ks.clone(), vec![])];
    if !ks.is_empty() {
        let zero: HashMap<ExprId, ExprId> = ks.iter().map(|&k| (k, pool.integer(0))).collect();
        let names: Vec<String> = ks
            .iter()
            .map(|&k| format!("{} = 0", pool.display(k)))
            .collect();
        tries.push((simp(sub_m(ode, &zero, pool), pool), vec![], names));
    }
    for (eq, consts, assume) in tries {
        let order = (0..ys.len())
            .rev()
            .find(|&k| contains(eq, ys[k], pool))
            .unwrap_or(0);
        if order == 0 {
            continue;
        }
        let input = crate::ode::dsolve::OdeInput {
            x: s,
            y: ys[0],
            derivs: ys[1..=order].to_vec(),
            equation: eq,
        };
        if let Ok(res) = crate::ode::dsolve::dsolve(&input, pool) {
            for sol in res.solutions {
                let mut c = consts.clone();
                c.extend(sol.constants.iter().copied());
                let mut st = steps.clone();
                st.push(format!(
                    "dsolve ({}): F = {}",
                    sol.method,
                    pool.display(sol.y_of_x)
                ));
                out.push(OdeSol {
                    f: sol.y_of_x,
                    constants: c,
                    assumptions: assume.clone(),
                    steps: st,
                });
            }
            if !out.is_empty() {
                break;
            }
        }
    }
    if out.is_empty() {
        // Separatrices of the autonomous second-order ODE (solitons, kinks).
        let zero: HashMap<ExprId, ExprId> = ks.iter().map(|&k| (k, pool.integer(0))).collect();
        let eq0 = simp(sub_m(ode, &zero, pool), pool);
        for mut sol in separatrices(eq0, ys, s, pool) {
            let mut st = steps.clone();
            st.append(&mut sol.steps);
            sol.steps = st;
            for &k in &ks {
                sol.assumptions
                    .insert(0, format!("{} = 0", pool.display(k)));
            }
            out.push(sol);
        }
    }
    out
}

/// `√r` with the square factors of a polynomial radicand pulled out.
fn sqrt_pulled(r: ExprId, y: ExprId, pool: &ExprPool) -> ExprId {
    let half = pool.rational(1, 2);
    let plain = pool.pow(r, half);
    // α + β cos y with α = ±β: 2α·cos²(y/2) or 2α·sin²(y/2).
    let cy = pool.func("cos", vec![y]);
    let cs = pool.symbol("__red_cos", Domain::Real);
    let r2 = simp(sub_m(r, &HashMap::from([(cy, cs)]), pool), pool);
    if !contains(r2, y, pool) {
        if let Some(beta) = d(r2, cs, pool) {
            let alpha = simp(
                sub_m(r2, &HashMap::from([(cs, pool.integer(0))]), pool),
                pool,
            );
            if !contains(beta, cs, pool) && !is_zero(beta, pool) {
                for (sign, trig) in [(1i64, "cos"), (-1, "sin")] {
                    if is_zero(
                        minus(alpha, pool.mul(vec![pool.integer(sign), beta]), pool),
                        pool,
                    ) {
                        let half_y = pool.mul(vec![half, y]);
                        return simp(
                            pool.mul(vec![
                                pool.pow(pool.mul(vec![pool.integer(2), alpha]), half),
                                pool.func(trig, vec![half_y]),
                            ]),
                            pool,
                        );
                    }
                }
            }
        }
    }
    let Ok(rf) = crate::poly::cancel::rational_function_of(r, vec![y], pool) else {
        return plain;
    };
    if rf.denom.total_degree() > 0 {
        return plain;
    }
    let Some((content, fs)) = rf.numer.factor_irreducible() else {
        return plain;
    };
    let mut outside = vec![];
    let mut inside = vec![pool.integer(content)];
    for (f, e) in fs {
        let fe = f.to_expr(pool);
        if e / 2 > 0 {
            outside.push(pool.pow(fe, pool.integer((e / 2) as i64)));
        }
        if e % 2 == 1 {
            inside.push(fe);
        }
    }
    let rden = rf.denom.to_expr(pool);
    inside.push(pool.pow(rden, pool.integer(-1)));
    outside.push(pool.pow(simp(pool.mul(inside), pool), half));
    simp(pool.mul(outside), pool)
}

/// Solutions of an autonomous `F'' = g(F)` on the separatrices through its
/// equilibria: `F'²/2 = G(F) − G(F*)`, `G' = g`, solved as first-order ODEs.
fn separatrices(ode: ExprId, ys: &[ExprId], s: ExprId, pool: &ExprPool) -> Vec<OdeSol> {
    let mut out = Vec::new();
    if contains(ode, s, pool) || (3..ys.len()).any(|k| contains(ode, ys[k], pool)) {
        return out;
    }
    let Some(top) = d(ode, ys[2], pool) else {
        return out;
    };
    if contains(top, ys[2], pool) || contains(top, ys[1], pool) || is_zero(top, pool) {
        return out;
    }
    let rest = simp(
        sub_m(ode, &HashMap::from([(ys[2], pool.integer(0))]), pool),
        pool,
    );
    if contains(rest, ys[1], pool) {
        return out;
    }
    let g = simp(
        pool.mul(vec![pool.integer(-1), quotient(rest, top, pool)]),
        pool,
    );
    let Some(gi) = super::solve::util::integrate(g, ys[0], pool) else {
        return out;
    };
    let num = match crate::poly::cancel::together_parts(g, vec![ys[0]], pool) {
        Ok((n, _)) => n,
        Err(_) => g,
    };
    let mut eqp: Vec<ExprId> = isolate(num, ys[0], pool)
        .into_iter()
        .filter(|&v| !contains(v, ys[0], pool))
        .map(|v| simp(v, pool))
        .collect();
    // Periodic nonlinearities: the equilibria 0, ±π are tried directly.
    let pi = pool.symbol("pi", Domain::Real);
    for cand in [pool.integer(0), pi, pool.mul(vec![pool.integer(-1), pi])] {
        if !eqp.contains(&cand) && is_zero(sub_m(g, &HashMap::from([(ys[0], cand)]), pool), pool) {
            eqp.push(cand);
        }
    }
    eqp.dedup();
    let debug = std::env::var("PDE_M6_DEBUG").is_ok();
    if debug {
        eprintln!(
            "separatrix: g = {}, G = {}, equilibria {:?}",
            pool.display(g),
            pool.display(gi),
            eqp.iter()
                .map(|&e| pool.display(e).to_string())
                .collect::<Vec<_>>()
        );
    }
    // No equilibrium (e^F): the separatrix of zero energy, F'² = 2G(F).
    let no_eq = eqp.is_empty();
    let mut levels: Vec<Option<ExprId>> = eqp.into_iter().take(4).map(Some).collect();
    if no_eq {
        levels.push(None);
    }
    for fstar_opt in levels {
        let (fstar, e0) = match fstar_opt {
            Some(f) => (f, simp(sub_m(gi, &HashMap::from([(ys[0], f)]), pool), pool)),
            None => (pool.integer(0), pool.integer(0)),
        };
        let label = if fstar_opt.is_some() {
            format!(
                "separatrix through the equilibrium F = {}",
                pool.display(fstar)
            )
        } else {
            "zero-energy separatrix F'^2 = 2G(F)".to_string()
        };
        let rad = crate::simplify::engine::simplify_expanded(
            pool.mul(vec![pool.integer(2), minus(gi, e0, pool)]),
            pool,
        )
        .value;
        let root = sqrt_pulled(rad, ys[0], pool);
        for sign in [1i64, -1] {
            let rhs = pool.mul(vec![pool.integer(sign), root]);
            // F' = A·sin(bF) → F = (2/b)·atan(exp(A b (s + C)));
            // F' = A·cos(bF) → F = (1/b)·asin(tanh(A b (s + C))).
            if let Some(v) = trig_autonomous(simp(rhs, pool), ys[0], s, pool) {
                out.push(OdeSol {
                    f: v.0,
                    constants: vec![v.1],
                    assumptions: vec![label.clone()],
                    steps: vec![format!(
                        "energy integral F'^2 = {} through F = {}",
                        pool.display(rad),
                        pool.display(fstar)
                    )],
                });
                continue;
            }
            let input = crate::ode::dsolve::OdeInput {
                x: s,
                y: ys[0],
                derivs: vec![ys[1]],
                equation: minus(ys[1], rhs, pool),
            };
            let ctx = crate::simplify::assumptions::AssumptionContext::new();
            let rep = crate::ode::dsolve::dsolve_with(&input, &ctx, pool);
            if debug {
                eprintln!(
                    "separatrix ODE F' = {}: {}",
                    pool.display(rhs),
                    match &rep {
                        Ok(r) => format!("{} branches", r.branches.len()),
                        Err(e) => format!("{e}"),
                    }
                );
            }
            let Ok(rep) = rep else {
                continue;
            };
            for br in rep.branches {
                let vals: Vec<ExprId> = match br.form {
                    crate::ode::dsolve::SolutionForm::Explicit(v) => vec![v],
                    crate::ode::dsolve::SolutionForm::Implicit(rel) => isolate(rel, ys[0], pool)
                        .into_iter()
                        .filter(|&v| !contains(v, ys[0], pool))
                        .collect(),
                };
                for v in vals {
                    out.push(OdeSol {
                        f: v,
                        constants: br.constants.clone(),
                        assumptions: vec![label.clone()],
                        steps: vec![format!(
                            "energy integral F'^2 = {} through F = {}",
                            pool.display(rad),
                            pool.display(fstar)
                        )],
                    });
                }
            }
        }
    }
    out
}

/// Whether a candidate is real-valued somewhere: it evaluates to a finite
/// real number at one of a few sample points of its free symbols.
fn real_somewhere(v: ExprId, pool: &ExprPool) -> bool {
    if contains(v, pool.imaginary_unit(), pool) {
        return false;
    }
    let syms: Vec<ExprId> = super::jet::symbols_of(v, pool).into_iter().collect();
    let mut any_eval = false;
    // Preserve the small probes and also sample outside the unit-scale box.
    // A real wave number can require |kappa| > 2 for its speed to be real.
    for k in 0..12 {
        let mut env = super::solve::util::sample_env(&syms, k % 6);
        let scale = if k < 6 { 1.0 } else { 4.0 };
        for value in env.values_mut() {
            *value *= if k % 2 == 0 { scale } else { -scale };
        }
        match crate::jit::eval_interp(v, &env, pool) {
            Some(x) if x.is_finite() => return true,
            Some(_) => any_eval = true,
            None => {}
        }
    }
    // Not evaluable at all (special functions the interpreter lacks): keep.
    !any_eval && !has_negative_radicand(v, &syms, pool)
}

fn has_negative_radicand(v: ExprId, syms: &[ExprId], pool: &ExprPool) -> bool {
    let mut stack = vec![v];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(n) {
            if let Some(q) = super::solve::util::as_rational(exp, pool) {
                if *q.denom() != 1 {
                    let env = super::solve::util::sample_env(syms, 0);
                    if let Some(b) = crate::jit::eval_interp(base, &env, pool) {
                        if b < 0.0 {
                            return true;
                        }
                    }
                }
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    false
}

fn verify_candidate(
    eq: ExprId,
    unknown: ExprId,
    value: ExprId,
    pool: &ExprPool,
) -> Option<PdeVerification> {
    let prob = PdeProblem::new(&[eq], &[unknown], pool).ok()?;
    let value = super::solve::tidy(simp(value, pool), pool);
    let debug = std::env::var("PDE_M6_DEBUG").is_ok();
    if !real_somewhere(value, pool) {
        if debug {
            eprintln!("candidate not real: {}", pool.display(value));
        }
        return None;
    }
    if debug {
        let v = verify_pde_solution(&prob, &[value], pool).ok()?;
        eprintln!(
            "candidate {} -> {} {:?}",
            pool.display(value),
            v.status.as_str(),
            v.checks
                .iter()
                .map(|c| pool.display(c.residual).to_string())
                .collect::<Vec<_>>()
        );
    }
    let v = upgrade_verification(verify_pde_solution(&prob, &[value], pool).ok()?, pool);
    (v.status == VerificationStatus::ExactlyVerified).then_some(v)
}

/// Re-run the zero test of the residuals that were not proved, after the
/// exact rewrites of [`expand_exact`]; upgrade to `exactly_verified` when
/// every residual is then proved zero.  Never downgrades, never certifies
/// on numeric evidence.
pub(crate) fn upgrade_verification(mut v: PdeVerification, pool: &ExprPool) -> PdeVerification {
    if !matches!(
        v.status,
        VerificationStatus::NumericallyChecked | VerificationStatus::Unverified
    ) {
        return v;
    }
    let mut all = true;
    for c in v.checks.iter_mut() {
        if c.status == crate::simplify::zero::ZeroStatus::ProvedZero {
            continue;
        }
        let r = simp(expand_exact(c.residual, pool), pool);
        let r = crate::simplify::engine::simplify_expanded(r, pool).value;
        let zt = crate::simplify::zero::zero_test(r, pool, None);
        if zt.status == crate::simplify::zero::ZeroStatus::ProvedZero {
            c.status = zt.status;
            c.method = "trig_of_atan_expansion";
            c.residual = r;
        } else {
            all = false;
        }
    }
    if all {
        v.status = VerificationStatus::ExactlyVerified;
    }
    v
}

/// Rewrite by exact identities: `(a^(p/q))^n → a^(pn/q)` when `q | n`, and
/// `sin/cos(k·atan(w) + R)` by the angle-addition formula with
/// `cos(kA) + i sin(kA) = (1 + i w)^k / (1 + w²)^(k/2)`.
pub(crate) fn expand_exact(e: ExprId, pool: &ExprPool) -> ExprId {
    match pool.get(e) {
        ExprData::Add(xs) => pool.add(xs.iter().map(|&x| expand_exact(x, pool)).collect()),
        ExprData::Mul(xs) => pool.mul(xs.iter().map(|&x| expand_exact(x, pool)).collect()),
        ExprData::Pow { base, exp } => {
            let b = expand_exact(base, pool);
            if let (ExprData::Pow { base: b2, exp: e2 }, Some(n)) =
                (pool.get(b), super::solve::util::as_rational(exp, pool))
            {
                if let Some(r) = super::solve::util::as_rational(e2, pool) {
                    // (a^r)^n = a^(rn) for an integer n.
                    let prod = rug::Rational::from(&r * &n);
                    if *n.denom() == 1 {
                        let (num, den) = prod.into_numer_denom();
                        return pool.pow(b2, pool.rational(num, den));
                    }
                }
            }
            pool.pow(b, expand_exact(exp, pool))
        }
        ExprData::Func { name, args } if (name == "sin" || name == "cos") && args.len() == 1 => {
            let arg = expand_exact(args[0], pool);
            let terms = match pool.get(arg) {
                ExprData::Add(xs) => xs,
                _ => vec![arg],
            };
            let mut hit = None;
            let mut rest = Vec::new();
            for t in terms {
                if hit.is_none() {
                    if let Some(kw) = k_atan(t, pool) {
                        hit = Some(kw);
                        continue;
                    }
                }
                rest.push(t);
            }
            let Some((k, w)) = hit else {
                return pool.func(name, vec![arg]);
            };
            let (ck, sk) = trig_of_k_atan(k, w, pool);
            if rest.is_empty() {
                return if name == "sin" { sk } else { ck };
            }
            let r = pool.add(rest);
            let (cr, sr) = (pool.func("cos", vec![r]), pool.func("sin", vec![r]));
            if name == "sin" {
                pool.add(vec![pool.mul(vec![sk, cr]), pool.mul(vec![ck, sr])])
            } else {
                pool.add(vec![
                    pool.mul(vec![ck, cr]),
                    pool.mul(vec![pool.integer(-1), sk, sr]),
                ])
            }
        }
        ExprData::Func { name, args } => {
            pool.func(name, args.iter().map(|&a| expand_exact(a, pool)).collect())
        }
        _ => e,
    }
}

/// `k·atan(w)` with an integer `k`.
fn k_atan(t: ExprId, pool: &ExprPool) -> Option<(i64, ExprId)> {
    let at = |x: ExprId| match pool.get(x) {
        ExprData::Func { name, args } if name == "atan" && args.len() == 1 => Some(args[0]),
        _ => None,
    };
    if let Some(w) = at(t) {
        return Some((1, w));
    }
    if let ExprData::Mul(xs) = pool.get(t) {
        if xs.len() == 2 {
            for (a, b) in [(xs[0], xs[1]), (xs[1], xs[0])] {
                if let (Some(w), ExprData::Integer(n)) = (at(b), pool.get(a)) {
                    let k = n.0.to_i64()?;
                    if k.abs() <= 8 && k != 0 {
                        return Some((k, w));
                    }
                }
            }
        }
    }
    None
}

/// `(cos(k·atan w), sin(k·atan w))`.
fn trig_of_k_atan(k: i64, w: ExprId, pool: &ExprPool) -> (ExprId, ExprId) {
    let n = k.unsigned_abs() as u32;
    let mut re = Vec::new();
    let mut im = Vec::new();
    let mut binom: i64 = 1;
    for j in 0..=n {
        if j > 0 {
            binom = binom * (n - j + 1) as i64 / j as i64;
        }
        let term = |c: i64| pool.mul(vec![pool.integer(c), pool.pow(w, pool.integer(j as i64))]);
        if j % 2 == 0 {
            let sign = if (j / 2) % 2 == 0 { 1 } else { -1 };
            re.push(term(sign * binom));
        } else {
            let sign = if ((j - 1) / 2) % 2 == 0 { 1 } else { -1 };
            im.push(term(sign * binom));
        }
    }
    let den = pool.pow(
        pool.add(vec![pool.integer(1), pool.pow(w, pool.integer(2))]),
        pool.rational(-(n as i64), 2),
    );
    let c = pool.mul(vec![pool.add(re), den]);
    let s = pool.mul(vec![pool.add(im), den]);
    let s = if k < 0 {
        pool.mul(vec![pool.integer(-1), s])
    } else {
        s
    };
    (c, s)
}

/// Replace a radical `√B(c)` of the speed by a new constant `κ` with
/// `B(c(κ)) = κ²`, so the family has no radical of its parameter.
fn clear_speed_radical(
    v: ExprId,
    c: ExprId,
    taken: &mut HashSet<String>,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let mut stack = vec![v];
    let mut seen = HashSet::new();
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        if let ExprData::Pow { base, exp } = pool.get(n) {
            if super::solve::util::as_rational(exp, pool) == Some(rug::Rational::from((1, 2)))
                && contains(base, c, pool)
                && super::jet::symbols_of(base, pool).len() == 1
            {
                let kappa = nice_symbol("kappa", taken, pool);
                let k2 = pool.pow(kappa, pool.integer(2));
                for cv in isolate(minus(base, k2, pool), c, pool) {
                    if contains(cv, c, pool) {
                        continue;
                    }
                    let back = sub_m(base, &HashMap::from([(c, cv)]), pool);
                    if !is_zero(minus(back, k2, pool), pool) {
                        continue;
                    }
                    let w = sub_m(v, &HashMap::from([(n, kappa)]), pool);
                    let w = sub_m(w, &HashMap::from([(c, cv)]), pool);
                    return Some((simp(w, pool), kappa));
                }
                return None;
            }
        }
        super::jet::push_children(n, pool, &mut stack);
    }
    None
}

fn assemble(red: &Red, sol: &OdeSol, pool: &ExprPool) -> ExprId {
    let fs = sub_m(sol.f, &HashMap::from([(red.s_sym, red.s_expr)]), pool);
    let v = sub_m(red.form, &HashMap::from([(red.ys[0], fs)]), pool);
    super::solve::tidy(simp(v, pool), pool)
}

fn push_unique(out: &mut Vec<ReducedSolution>, r: ReducedSolution, pool: &ExprPool) {
    if out
        .iter()
        .any(|o| is_zero(minus(o.value, r.value, pool), pool))
    {
        return;
    }
    out.push(r);
}

// ---------------------------------------------------------------------------
// Public API: reductions
// ---------------------------------------------------------------------------

fn setup(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<(Jets, ExprId, HashSet<String>), PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    let mut taken = names_in(&[eq, f], pool);
    for x in &jets.indep {
        taken.insert(pool.display(*x).to_string());
    }
    Ok((jets, f, taken))
}

/// The reduction of `eq = 0` by the generator with components
/// `(ξ_x, ξ_t, η)` (in the independent variables and the symbol `u`).
///
/// # Errors
///
/// `E-PDE-302` when the invariants cannot be found or the reduced equation
/// still depends on the eliminated variable.
pub fn reduce_by_generator(
    eq: ExprId,
    unknown: ExprId,
    components: &[ExprId],
    pool: &ExprPool,
) -> Result<Reduction, PdeError> {
    let (jets, f, mut taken) = setup(eq, unknown, pool)?;
    let comps: Vec<ExprId> = components
        .iter()
        .map(|&c| jets.to_jets(c, pool))
        .collect::<Result<_, _>>()?;
    let red = reduce(&jets, f, &comps, &mut taken, "s", pool)?;
    Ok(public_reduction(&red, &mut taken, pool))
}

fn public_reduction(red: &Red, taken: &mut HashSet<String>, pool: &ExprPool) -> Reduction {
    let fname = if taken.insert("F".into()) {
        "F".to_string()
    } else {
        pool.display(fresh_symbol("F", taken, pool)).to_string()
    };
    let fs = pool.func(fname.clone(), vec![red.s_sym]);
    let mut m = HashMap::new();
    for (k, &y) in red.ys.iter().enumerate() {
        let v = if k == 0 {
            fs
        } else {
            crate::diff::undefined::build(fs, &[k as u32], pool)
        };
        m.insert(y, v);
    }
    let ode = simp(sub_m(red.ode, &m, pool), pool);
    let fx = pool.func(fname, vec![red.s_expr]);
    let form = simp(
        sub_m(red.form, &HashMap::from([(red.ys[0], fx)]), pool),
        pool,
    );
    Reduction {
        variable: red.s_expr,
        symbol: red.s_sym,
        form,
        ode,
        function: fs,
    }
}

/// Traveling-wave solutions `u = F(x − ct)` of `eq = 0` (two independent
/// variables; `t` is the one named `t`, else the second): general solutions
/// of the reduced ODE by exact integration and `dsolve`, then the tanh /
/// sech / tan expansion methods (special speeds such as the Fisher–KPP
/// fronts).  Every solution is verified.
///
/// # Errors
///
/// `E-PDE-302` when the equation is not autonomous in `x, t`; `E-PDE-303`
/// when no candidate verified.
pub fn traveling_wave_solutions(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ReducedSolution>, PdeError> {
    let (jets, f, mut taken) = setup(eq, unknown, pool)?;
    if jets.p() != 2 {
        return Err(PdeError::ReductionFailed(
            "traveling waves need two independent variables".into(),
        ));
    }
    let ti = jets
        .indep
        .iter()
        .position(|&x| pool.display(x).to_string() == "t")
        .unwrap_or(1);
    let c = nice_symbol("c", &mut taken, pool);
    let mut comps = vec![pool.integer(0); 3];
    comps[1 - ti] = c;
    comps[ti] = pool.integer(1);
    let red = reduce(&jets, f, &comps, &mut taken, "xi", pool)?;
    let mut out = Vec::new();
    for sol in capped(REDUCE_SECONDS, || solve_reduced(&red, &mut taken, pool)) {
        let mut v = assemble(&red, &sol, pool);
        let mut speed = c;
        if let Some((w, kappa)) = clear_speed_radical(v, c, &mut taken, pool) {
            v = w;
            speed = kappa;
        }
        if let Some(ver) = verify_candidate(eq, unknown, v, pool) {
            let mut consts = vec![speed];
            consts.extend(sol.constants.iter().copied());
            let mut steps = red.steps.clone();
            steps.extend(sol.steps.iter().cloned());
            push_unique(
                &mut out,
                ReducedSolution {
                    value: simp(v, pool),
                    method: "traveling_wave",
                    constants: consts,
                    assumptions: sol.assumptions.clone(),
                    steps,
                    verification: ver,
                },
                pool,
            );
        }
    }
    for basis in [Basis::Tanh, Basis::Sech, Basis::Tan] {
        let cands = capped(ANSATZ_SECONDS, || {
            ansatz_solutions(&red, c, basis, &mut taken, pool)
        });
        for (v, consts, steps) in cands {
            if let Some(ver) = verify_candidate(eq, unknown, v, pool) {
                push_unique(
                    &mut out,
                    ReducedSolution {
                        value: simp(v, pool),
                        method: basis.name(),
                        constants: consts,
                        assumptions: Vec::new(),
                        steps,
                        verification: ver,
                    },
                    pool,
                );
            }
        }
    }
    if out.is_empty() {
        return Err(PdeError::NoReducedSolution(format!(
            "the traveling-wave ODE {} = 0 has no solution that verified",
            pool.display(red.ode)
        )));
    }
    Ok(out)
}

/// Solutions of `eq = 0` invariant under the generator `(ξ_x, ξ_t, η)`.
///
/// # Errors
///
/// As [`reduce_by_generator`]; `E-PDE-303` when nothing verified.
pub fn invariant_solutions(
    eq: ExprId,
    unknown: ExprId,
    components: &[ExprId],
    pool: &ExprPool,
) -> Result<Vec<ReducedSolution>, PdeError> {
    let (jets, f, mut taken) = setup(eq, unknown, pool)?;
    let comps: Vec<ExprId> = components
        .iter()
        .map(|&c| jets.to_jets(c, pool))
        .collect::<Result<_, _>>()?;
    let red = reduce(&jets, f, &comps, &mut taken, "s", pool)?;
    let out = solutions_of(&red, eq, unknown, "invariant", &mut taken, pool);
    if out.is_empty() {
        return Err(PdeError::NoReducedSolution(format!(
            "the reduced ODE {} = 0 has no solution that verified",
            pool.display(red.ode)
        )));
    }
    Ok(out)
}

fn solutions_of(
    red: &Red,
    eq: ExprId,
    unknown: ExprId,
    method: &'static str,
    taken: &mut HashSet<String>,
    pool: &ExprPool,
) -> Vec<ReducedSolution> {
    let mut out = Vec::new();
    for sol in capped(REDUCE_SECONDS, || solve_reduced(red, taken, pool)) {
        let v = assemble(red, &sol, pool);
        if let Some(ver) = verify_candidate(eq, unknown, v, pool) {
            let mut steps = red.steps.clone();
            steps.extend(sol.steps.iter().cloned());
            push_unique(
                &mut out,
                ReducedSolution {
                    value: simp(v, pool),
                    method,
                    constants: sol.constants.clone(),
                    assumptions: sol.assumptions.clone(),
                    steps,
                    verification: ver,
                },
                pool,
            );
        }
    }
    out
}

/// Similarity solutions: reductions by every finite point-symmetry
/// generator that moves `x` or `t`, and by combinations `X + λY` whose
/// reduced ODE is exact for some `λ` (Barenblatt, heat source).  Every
/// solution is verified.
///
/// # Errors
///
/// As [`super::point_symmetries`]; `E-PDE-303` when nothing verified.
pub fn similarity_solutions(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<ReducedSolution>, PdeError> {
    let cases = super::symmetry::point_symmetries(eq, unknown, pool)?;
    let Some(case) = cases.into_iter().find(|c| c.case.is_empty()) else {
        return Err(PdeError::ReductionFailed("no generic symmetry case".into()));
    };
    let (jets, f, mut taken) = setup(eq, unknown, pool)?;
    if jets.p() != 2 {
        return Err(PdeError::ReductionFailed(
            "similarity reductions need two independent variables".into(),
        ));
    }
    let gens: Vec<Vec<ExprId>> = case.generators.iter().map(|g| g.components()).collect();
    let moves = |g: &[ExprId]| !(is_zero(g[0], pool) && is_zero(g[1], pool));
    let mut out = Vec::new();
    let mut tried = 0;
    for g in gens.iter().filter(|g| moves(g)) {
        if tried > 24 || crate::budget::tripped().is_some() {
            break;
        }
        tried += 1;
        if let Ok(red) = reduce(&jets, f, g, &mut taken, "s", pool) {
            for r in solutions_of(&red, eq, unknown, "similarity", &mut taken, pool) {
                push_unique(&mut out, r, pool);
            }
        }
    }
    // X + λ Y with λ from exactness of the reduced ODE.
    let lam = fresh_symbol("__lam", &mut taken, pool);
    for (i, gx) in gens.iter().enumerate() {
        for (j, gy) in gens.iter().enumerate() {
            if i == j || tried > 40 || crate::budget::tripped().is_some() {
                continue;
            }
            let comb: Vec<ExprId> = gx
                .iter()
                .zip(gy)
                .map(|(&a, &b)| simp(pool.add(vec![a, pool.mul(vec![lam, b])]), pool))
                .collect();
            if !moves(&comb) {
                continue;
            }
            // Only scaling-type combinations (non-constant ξ) are of interest.
            if gx[..2].iter().all(|&c| is_number(c, pool)) {
                continue;
            }
            tried += 1;
            let Ok(red) = reduce(&jets, f, &comb, &mut taken, "s", pool) else {
                continue;
            };
            if !contains(red.ode, lam, pool) {
                continue;
            }
            for v in exactness_values(red.ode, lam, red.s_sym, &red.ys, pool) {
                let m = HashMap::from([(lam, v)]);
                let red2 = Red {
                    s_expr: simp(sub_m(red.s_expr, &m, pool), pool),
                    form: simp(sub_m(red.form, &m, pool), pool),
                    ode: simp(sub_m(red.ode, &m, pool), pool),
                    ..red.clone()
                };
                for r in solutions_of(&red2, eq, unknown, "similarity", &mut taken, pool) {
                    push_unique(&mut out, r, pool);
                }
            }
        }
    }
    if out.is_empty() {
        return Err(PdeError::NoReducedSolution(
            "no similarity reduction produced a verified solution".into(),
        ));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Expansion (ansatz) methods
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Basis {
    Tanh,
    Sech,
    Tan,
}

impl Basis {
    fn name(self) -> &'static str {
        match self {
            Basis::Tanh => "tanh",
            Basis::Sech => "sech",
            Basis::Tan => "tan",
        }
    }
}

/// The pole order `N` from balancing the highest derivative against the
/// strongest nonlinearity (each derivative raises the degree in `Y` by one).
#[cfg_attr(not(feature = "groebner"), allow(dead_code))]
fn balance(ode: ExprId, ys: &[ExprId], pool: &ExprPool) -> Vec<u32> {
    let ex = crate::simplify::engine::simplify_expanded(ode, pool).value;
    let terms = match pool.get(ex) {
        ExprData::Add(xs) => xs,
        _ => vec![ex],
    };
    // Each term: (coefficient of N, constant) in its degree.
    let mut lin: Vec<(u32, u32)> = Vec::new();
    for t in terms {
        let mut a = 0u32;
        let mut b = 0u32;
        let factors = match pool.get(t) {
            ExprData::Mul(xs) => xs,
            _ => vec![t],
        };
        for fct in factors {
            let (base, e) = match pool.get(fct) {
                ExprData::Pow { base, exp } => match pool.get(exp) {
                    ExprData::Integer(n) => (base, n.0.to_u32().unwrap_or(0)),
                    _ => (fct, 0),
                },
                _ => (fct, 1),
            };
            if let Some(k) = ys.iter().position(|&y| y == base) {
                a += e;
                b += e * k as u32;
            }
        }
        if a > 0 {
            lin.push((a, b));
        }
    }
    let mut out = Vec::new();
    for n in 1..=4u32 {
        let degs: Vec<(u32, (u32, u32))> = lin.iter().map(|&(a, b)| (a * n + b, (a, b))).collect();
        let max = degs.iter().map(|x| x.0).max().unwrap_or(0);
        let tops: HashSet<(u32, u32)> = degs.iter().filter(|x| x.0 == max).map(|x| x.1).collect();
        let distinct_a: HashSet<u32> = tops.iter().map(|x| x.0).collect();
        if distinct_a.len() >= 2 {
            out.push(n);
        }
    }
    out
}

/// A polynomial term: exponent vector and integer coefficient.
#[cfg(feature = "groebner")]
type Term = (Vec<u32>, rug::Integer);

#[cfg(feature = "groebner")]
fn ansatz_solutions(
    red: &Red,
    c: ExprId,
    basis: Basis,
    taken: &mut HashSet<String>,
    pool: &ExprPool,
) -> Vec<(ExprId, Vec<ExprId>, Vec<String>)> {
    use crate::solver::{solve_polynomial_system, SolutionSet};
    let ys = &red.ys;
    if contains(red.ode, red.s_sym, pool) {
        return vec![];
    }
    // Polynomial in the Y's?
    let Ok((rf, gens)) = crate::poly::cancel::rational_over_generators(red.ode, ys.clone(), pool)
    else {
        return vec![];
    };
    if gens[ys.len()..]
        .iter()
        .any(|&g| ys.iter().any(|&y| contains(g, y, pool)))
    {
        return vec![];
    }
    let _ = rf;
    let ns: Vec<u32> = match basis {
        Basis::Sech => vec![1, 2],
        _ => balance(red.ode, ys, pool),
    };
    let y = fresh_symbol("__Y", taken, pool);
    let z = fresh_symbol("__Z", taken, pool);
    let k = nice_symbol("k", taken, pool);
    let mut out = Vec::new();
    for n in ns {
        if crate::budget::tripped().is_some() {
            break;
        }
        let a: Vec<ExprId> = (0..=n)
            .map(|i| fresh_symbol(&format!("__a{i}_"), taken, pool))
            .collect();
        // f = Σ a_i Y^i and its derivatives.
        let f0 = pool.add(
            (0..=n as usize)
                .map(|i| pool.mul(vec![a[i], pool.pow(y, pool.integer(i as i64))]))
                .collect(),
        );
        let dd = |g: ExprId| -> ExprId {
            let gy = d(g, y, pool).unwrap_or_else(|| pool.integer(0));
            let one = pool.integer(1);
            let ysq = pool.pow(y, pool.integer(2));
            match basis {
                Basis::Tanh => simp(pool.mul(vec![gy, k, minus(one, ysq, pool)]), pool),
                Basis::Tan => simp(pool.mul(vec![gy, k, pool.add(vec![one, ysq])]), pool),
                Basis::Sech => {
                    let gz = d(g, z, pool).unwrap_or_else(|| pool.integer(0));
                    simp(
                        pool.add(vec![
                            pool.mul(vec![pool.integer(-1), gy, k, y, z]),
                            pool.mul(vec![gz, k, ysq]),
                        ]),
                        pool,
                    )
                }
            }
        };
        let mut m = HashMap::new();
        let mut cur = f0;
        for (i, &yy) in ys.iter().enumerate().take(red.order + 1) {
            if i > 0 {
                cur = dd(cur);
            }
            m.insert(yy, cur);
        }
        let e = simp(sub_m(red.ode, &m, pool), pool);
        let Ok((rf, gens)) = crate::poly::cancel::rational_over_generators(e, vec![y, z], pool)
        else {
            continue;
        };
        // Reduce Z² → 1 − Y² (sech basis), collect coefficients of Y^i Z^j.
        let mut coeffs: HashMap<(u32, u32), Vec<Term>> = HashMap::new();
        let mut work: Vec<Term> = rf
            .numer
            .terms
            .iter()
            .map(|(e, c)| {
                let mut e = e.clone();
                e.resize(gens.len(), 0);
                (e, c.clone())
            })
            .collect();
        let mut guard = 0;
        while let Some((e, c)) = work.pop() {
            guard += 1;
            if guard > 100_000 {
                break;
            }
            if e[1] >= 2 {
                let mut a1 = e.clone();
                a1[1] -= 2;
                let mut a2 = a1.clone();
                a2[0] += 2;
                work.push((a1, c.clone()));
                work.push((a2, -c));
                continue;
            }
            let mut rest = e.clone();
            rest[0] = 0;
            rest[1] = 0;
            coeffs.entry((e[0], e[1])).or_default().push((rest, c));
        }
        let mut eqs = Vec::new();
        for (_, ts) in coeffs {
            let mut tm: std::collections::BTreeMap<Vec<u32>, rug::Integer> =
                std::collections::BTreeMap::new();
            for (mut k2, c2) in ts {
                while k2.last() == Some(&0) {
                    k2.pop();
                }
                *tm.entry(k2).or_insert_with(|| rug::Integer::from(0)) += c2;
            }
            tm.retain(|_, c2| *c2 != 0);
            if tm.is_empty() {
                continue;
            }
            let p = crate::poly::multipoly::MultiPoly {
                vars: gens.clone(),
                terms: tm,
            }
            .to_expr(pool);
            eqs.push(p);
        }
        if eqs.is_empty() {
            continue;
        }
        let kk = fresh_symbol("__kinv", taken, pool);
        let ai = fresh_symbol("__ainv", taken, pool);
        let nz_k = minus(pool.mul(vec![k, kk]), pool.integer(1), pool);
        let nz_a = minus(pool.mul(vec![a[n as usize], ai]), pool.integer(1), pool);
        let attempts: Vec<(Vec<ExprId>, Vec<ExprId>)> = vec![
            {
                let mut u = a.clone();
                u.extend([k, c, kk, ai]);
                (u, vec![nz_k, nz_a])
            },
            {
                let mut u = a.clone();
                u.extend([c, ai]);
                (u, vec![nz_a])
            },
            {
                let mut u = a[1..].to_vec();
                u.extend([c, ai]);
                (u, vec![nz_a])
            },
            {
                let mut u = a.clone();
                u.extend([k, kk, ai]);
                (u, vec![nz_k, nz_a])
            },
        ];
        let mut found_any = false;
        for (unknowns, extra) in attempts {
            if found_any || crate::budget::tripped().is_some() {
                break;
            }
            let mut sys = eqs.clone();
            sys.extend(extra);
            let res = solve_polynomial_system(sys.clone(), unknowns.clone(), pool);
            if std::env::var("PDE_M6_DEBUG").is_ok() {
                eprintln!(
                    "ansatz {} n={n} unknowns={} eqs={:?} -> {}",
                    basis.name(),
                    unknowns.len(),
                    sys.iter()
                        .map(|&q| pool.display(q).to_string())
                        .collect::<Vec<_>>(),
                    match &res {
                        Ok(SolutionSet::Finite(v)) => format!("finite {}", v.len()),
                        Ok(SolutionSet::Parametric(_)) => "parametric".into(),
                        Ok(SolutionSet::NoSolution) => "none".into(),
                        Err(e) => format!("err {e:?}"),
                    }
                );
            }
            let Ok(SolutionSet::Finite(sols)) = res else {
                let _ = crate::solver::take_solve_side_conditions();
                continue;
            };
            let _ = crate::solver::take_solve_side_conditions();
            for sol in sols {
                let mut mm: HashMap<ExprId, ExprId> = HashMap::new();
                for (&u, &v) in unknowns.iter().zip(&sol) {
                    mm.insert(u, v);
                }
                // Skip mirror images k < 0 of a numeric wavenumber.
                if let Some(&kv) = mm.get(&k) {
                    if let Some(x) = super::solve::util::eval_at(kv, &HashMap::new(), pool) {
                        if x < 0.0 {
                            continue;
                        }
                    }
                }
                let xi = sub_m(red.s_expr, &mm, pool);
                let kv = mm.get(&k).copied().unwrap_or(k);
                let arg = pool.mul(vec![kv, xi]);
                let yv = match basis {
                    Basis::Tanh => pool.func("tanh", vec![arg]),
                    Basis::Tan => pool.func("tan", vec![arg]),
                    Basis::Sech => pool.pow(pool.func("cosh", vec![arg]), pool.integer(-1)),
                };
                let fv = sub_m(f0, &mm, pool);
                let fv = sub_m(fv, &HashMap::from([(y, yv)]), pool);
                let v = sub_m(red.form, &HashMap::from([(red.ys[0], fv)]), pool);
                let mut v = simp(v, pool);
                if contains(v, z, pool) {
                    continue;
                }
                // Coefficients left free become named constants.
                let mut consts = Vec::new();
                for (i, &ai2) in a.iter().enumerate() {
                    if contains(v, ai2, pool) {
                        let nm = nice_symbol(&format!("A{i}"), taken, pool);
                        v = simp(sub_m(v, &HashMap::from([(ai2, nm)]), pool), pool);
                        consts.push(nm);
                    }
                }
                for sym in [c, k] {
                    if contains(v, sym, pool) {
                        consts.push(sym);
                    }
                }
                found_any = true;
                out.push((
                    v,
                    consts,
                    vec![format!(
                        "{} expansion of order {n} on the traveling-wave ODE; coefficient system \
                         solved by Gröbner bases",
                        basis.name()
                    )],
                ));
            }
        }
    }
    out
}

#[cfg(not(feature = "groebner"))]
fn ansatz_solutions(
    _red: &Red,
    _c: ExprId,
    _basis: Basis,
    _taken: &mut HashSet<String>,
    _pool: &ExprPool,
) -> Vec<(ExprId, Vec<ExprId>, Vec<String>)> {
    Vec::new()
}

/// Expansion-method solutions `u = Σ aᵢ Y(k(x − ct))ⁱ` of `eq = 0` with
/// `Y = tanh`, `sech` or `tan` (`method`), the pole order found by
/// balancing; the coefficient system is solved with Gröbner bases and every
/// solution verified.
///
/// # Errors
///
/// `E-PDE-302` for an equation that is not autonomous or not polynomial;
/// `E-PDE-303` when nothing verified; `E-PDE-050` for an unknown method.
pub fn ansatz_solutions_for(
    eq: ExprId,
    unknown: ExprId,
    method: &str,
    pool: &ExprPool,
) -> Result<Vec<ReducedSolution>, PdeError> {
    let basis = match method {
        "tanh" => Basis::Tanh,
        "sech" => Basis::Sech,
        "tan" => Basis::Tan,
        other => {
            return Err(PdeError::PdsolveUnsupported(format!(
                "unknown expansion method {other:?} (tanh, sech, tan)"
            )))
        }
    };
    let (jets, f, mut taken) = setup(eq, unknown, pool)?;
    if jets.p() != 2 {
        return Err(PdeError::ReductionFailed(
            "expansion methods need two independent variables".into(),
        ));
    }
    let ti = jets
        .indep
        .iter()
        .position(|&x| pool.display(x).to_string() == "t")
        .unwrap_or(1);
    let c = nice_symbol("c", &mut taken, pool);
    let mut comps = vec![pool.integer(0); 3];
    comps[1 - ti] = c;
    comps[ti] = pool.integer(1);
    let red = reduce(&jets, f, &comps, &mut taken, "xi", pool)?;
    let mut out = Vec::new();
    for (v, consts, steps) in ansatz_solutions(&red, c, basis, &mut taken, pool) {
        if let Some(ver) = verify_candidate(eq, unknown, v, pool) {
            push_unique(
                &mut out,
                ReducedSolution {
                    value: simp(v, pool),
                    method: basis.name(),
                    constants: consts,
                    assumptions: Vec::new(),
                    steps,
                    verification: ver,
                },
                pool,
            );
        }
    }
    if out.is_empty() {
        return Err(PdeError::NoReducedSolution(format!(
            "no {} expansion verified",
            basis.name()
        )));
    }
    Ok(out)
}

/// `F' = A·sin(bF)` / `A·cos(bF)` (`A`, `b` free of `F`): the closed-form
/// solution and its constant.
fn trig_autonomous(rhs: ExprId, f: ExprId, s: ExprId, pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    let factors = match pool.get(rhs) {
        ExprData::Mul(xs) => xs,
        _ => vec![rhs],
    };
    let mut trig = None;
    let mut rest = Vec::new();
    for x in factors {
        match pool.get(x) {
            ExprData::Func { name, args }
                if (name == "sin" || name == "cos") && args.len() == 1 && trig.is_none() =>
            {
                trig = Some((name.clone(), args[0]));
            }
            _ => rest.push(x),
        }
    }
    let (name, arg) = trig?;
    let a = simp(pool.mul(rest), pool);
    if contains(a, f, pool) {
        return None;
    }
    let b = d(arg, f, pool)?;
    if contains(b, f, pool)
        || !is_zero(
            sub_m(arg, &HashMap::from([(f, pool.integer(0))]), pool),
            pool,
        )
    {
        return None;
    }
    let c = pool.symbol("C1", Domain::Real);
    let z = pool.mul(vec![a, b, pool.add(vec![s, c])]);
    let inv_b = pool.pow(b, pool.integer(-1));
    let v = if name == "sin" {
        pool.mul(vec![
            pool.integer(2),
            inv_b,
            pool.func("atan", vec![pool.func("exp", vec![z])]),
        ])
    } else {
        // tan(G/2 + π/4) = e^z  ⇒  G = 2·atan(e^z) − π/2.
        let pi = pool.symbol("pi", Domain::Real);
        pool.mul(vec![
            inv_b,
            pool.add(vec![
                pool.mul(vec![
                    pool.integer(2),
                    pool.func("atan", vec![pool.func("exp", vec![z])]),
                ]),
                pool.mul(vec![pool.rational(-1, 2), pi]),
            ]),
        ])
    };
    Some((simp(v, pool), c))
}
