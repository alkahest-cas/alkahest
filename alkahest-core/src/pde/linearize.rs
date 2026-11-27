//! Linearising and Bäcklund-type transformations (milestone M6): Cole–Hopf
//! (Burgers → heat), Miura (mKdV → KdV), the hodograph transform of a 2×2
//! quasilinear first-order system, and the Hopf–Lax formula for
//! Hamilton–Jacobi equations.
//!
//! Every transformation is **verified as an identity on jets**: Cole–Hopf by
//! reducing the transformed Burgers residual modulo the heat equation and its
//! derivatives, Miura by checking `KdV[u] = (2αv + β D_x) mKdV[v]`, the
//! hodograph by the inverse-function relations, Hopf–Lax by the Legendre
//! duality `L'(H'(p)) = p`.

use std::collections::HashMap;

use crate::kernel::{Domain, ExprId, ExprPool};

use super::error::PdeError;
use super::jet::JetSpace;
use super::solve::util::{contains, d, is_number, is_zero, isolate, simp};
use super::symmetry::{Jets, Reducer};

fn minus(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, pool.mul(vec![pool.integer(-1), b])])
}

fn quotient(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, pool.pow(b, pool.integer(-1))])
}

fn time_space(jets: &Jets, pool: &ExprPool) -> Result<(usize, usize), PdeError> {
    if jets.p() != 2 {
        return Err(PdeError::TransformNotApplicable(
            "the transformation needs two independent variables (x, t)".into(),
        ));
    }
    let ti = jets
        .indep
        .iter()
        .position(|&x| pool.display(x).to_string() == "t")
        .unwrap_or(1);
    Ok((1 - ti, ti))
}

fn alpha(p: usize, i: usize, k: u32) -> Vec<u32> {
    let mut a = vec![0; p];
    a[i] = k;
    a
}

/// The Cole–Hopf linearisation of a Burgers equation.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ColeHopf {
    /// `u = −(2ν/a)·θ_x/θ` (with derivative nodes of `θ(x, t)`).
    pub substitution: ExprId,
    /// The heat equation `θ_t − ν θ_xx` (`= 0`).
    pub heat_equation: ExprId,
    /// The new unknown `θ(x, t)`.
    pub theta: ExprId,
    /// `ν`.
    pub viscosity: ExprId,
    /// `a` in `u_t + a u u_x = ν u_xx`.
    pub coefficient: ExprId,
    /// `Burgers[u(θ)] ≡ 0` modulo the heat equation, proved exactly.
    pub verified: bool,
}

/// Match `λ(u_t + a u u_x − ν u_xx)` with constant `a, ν ≠ 0`.
fn burgers_coefficients(
    jets: &Jets,
    f: ExprId,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId, usize, usize), PdeError> {
    let (xi, ti) = time_space(jets, pool)?;
    let p = 2;
    let ut = jets.jet(&alpha(p, ti, 1), pool);
    let ux = jets.jet(&alpha(p, xi, 1), pool);
    let uxx = jets.jet(&alpha(p, xi, 2), pool);
    let lam = d(f, ut, pool).unwrap_or_else(|| pool.integer(0));
    let not_burgers = || {
        PdeError::TransformNotApplicable(format!(
            "{} is not of Burgers form u_t + a u u_x = nu u_xx",
            pool.display(f)
        ))
    };
    if is_zero(lam, pool) || !is_constant(lam, jets, pool) {
        return Err(not_burgers());
    }
    let a = simp(
        quotient(
            d(d(f, ux, pool).ok_or_else(not_burgers)?, jets.w, pool).ok_or_else(not_burgers)?,
            lam,
            pool,
        ),
        pool,
    );
    let nu = simp(
        pool.mul(vec![
            pool.integer(-1),
            quotient(d(f, uxx, pool).ok_or_else(not_burgers)?, lam, pool),
        ]),
        pool,
    );
    if is_zero(a, pool)
        || is_zero(nu, pool)
        || !is_constant(a, jets, pool)
        || !is_constant(nu, jets, pool)
    {
        return Err(not_burgers());
    }
    let model = pool.add(vec![
        ut,
        pool.mul(vec![a, jets.w, ux]),
        pool.mul(vec![pool.integer(-1), nu, uxx]),
    ]);
    if !is_zero(minus(f, pool.mul(vec![lam, model]), pool), pool) {
        return Err(not_burgers());
    }
    Ok((a, nu, xi, ti))
}

fn is_constant(e: ExprId, jets: &Jets, pool: &ExprPool) -> bool {
    !jets.indep.iter().any(|&x| contains(e, x, pool)) && jets.jets_in(e, pool).is_empty()
}

/// The Cole–Hopf transformation of `u_t + a u u_x = ν u_xx` (constant
/// `a, ν`): `u = −(2ν/a) θ_x/θ` with `θ_t = ν θ_xx`, verified on jets.
///
/// # Errors
///
/// `E-PDE-304` when the equation is not of Burgers form.
pub fn cole_hopf(eq: ExprId, unknown: ExprId, pool: &ExprPool) -> Result<ColeHopf, PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    let (a, nu, xi, ti) = burgers_coefficients(&jets, f, pool)?;
    let mut taken = super::solve::util::names_in(&[eq], pool);
    let name = if taken.insert("theta".into()) {
        "theta".to_string()
    } else {
        pool.display(super::solve::util::fresh_symbol("theta", &mut taken, pool))
            .to_string()
    };
    let theta = pool.func(name, jets.indep.clone());
    let tj = Jets::new(theta, pool)?;
    // u in θ-jets.
    let tx = tj.jet(&alpha(2, xi, 1), pool);
    let coef = simp(
        pool.mul(vec![pool.integer(-2), nu, pool.pow(a, pool.integer(-1))]),
        pool,
    );
    let u_of = simp(
        pool.mul(vec![coef, tx, pool.pow(tj.w, pool.integer(-1))]),
        pool,
    );
    let mut m = HashMap::new();
    m.insert(jets.w, u_of);
    for (al, s) in jets.jets_in(f, pool) {
        if al.iter().sum::<u32>() == 0 {
            continue;
        }
        m.insert(s, tj.total_d_multi(u_of, &al, None, pool));
    }
    let r = simp(crate::kernel::subs::subs(f, &m, pool), pool);
    // Reduce modulo θ_t = ν θ_xx.
    let heat_j = minus(
        tj.jet(&alpha(2, ti, 1), pool),
        pool.mul(vec![nu, tj.jet(&alpha(2, xi, 2), pool)]),
        pool,
    );
    let mut red = Reducer::new(&tj, heat_j, pool)?;
    let rr = red.reduce(r, &tj, pool);
    let rr = crate::simplify::engine::simplify_expanded(rr, pool).value;
    let verified = is_zero(rr, pool);
    let space = JetSpace::new(&tj.indep, &[theta], 4, pool)?;
    Ok(ColeHopf {
        substitution: space.from_jets(u_of, pool),
        heat_equation: space.from_jets(heat_j, pool),
        theta,
        viscosity: nu,
        coefficient: a,
        verified,
    })
}

/// Exact Burgers solutions from heat solutions `θ = 1 + Σ exp(kᵢx + νkᵢ²t + δᵢ)`
/// (one and two fronts), through Cole–Hopf.  Candidates; the caller verifies.
pub(crate) fn cole_hopf_families(
    eq: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<Vec<(ExprId, Vec<ExprId>)>, PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    let (a, nu, xi, ti) = burgers_coefficients(&jets, f, pool)?;
    let (x, t) = (jets.indep[xi], jets.indep[ti]);
    let mut taken = super::solve::util::names_in(&[eq], pool);
    let mut out = Vec::new();
    let coef = pool.mul(vec![pool.integer(-2), nu, pool.pow(a, pool.integer(-1))]);
    for n in 1..=2 {
        let mut consts = Vec::new();
        let mut th = vec![pool.integer(1)];
        for _ in 0..n {
            let k = super::solve::util::fresh_symbol("k", &mut taken, pool);
            let dl = super::solve::util::fresh_symbol("delta", &mut taken, pool);
            consts.push(k);
            consts.push(dl);
            th.push(pool.func(
                "exp",
                vec![pool.add(vec![
                    pool.mul(vec![k, x]),
                    pool.mul(vec![nu, pool.pow(k, pool.integer(2)), t]),
                    dl,
                ])],
            ));
        }
        let theta = pool.add(th);
        let thx = d(theta, x, pool).unwrap_or_else(|| pool.integer(0));
        let u = simp(
            pool.mul(vec![coef, thx, pool.pow(theta, pool.integer(-1))]),
            pool,
        );
        out.push((u, consts));
    }
    Ok(out)
}

/// The Miura map between mKdV and KdV.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Miura {
    /// `u = α v² + β v_x` (derivative nodes of `v`).
    pub substitution: ExprId,
    /// The KdV equation satisfied by `u` (`u_t + A u u_x + b u_xxx`, in jets
    /// of the KdV unknown).
    pub kdv: ExprId,
    /// The KdV unknown.
    pub kdv_unknown: ExprId,
    /// `KdV[u(v)] = (2αv + β D_x) mKdV[v]` proved exactly.
    pub verified: bool,
}

/// The Miura transformation of `v_t + a v² v_x + b v_xxx = 0` (constant
/// `a, b`): `u = α v² + β v_x` solves `u_t + A u u_x + b u_xxx = 0`, with
/// `α, β, A` found from the identity `KdV[u] = (2αv + β D_x)·mKdV[v]` and
/// that identity verified.  `kdv_unknown` names the KdV unknown `u(x, t)`.
///
/// # Errors
///
/// `E-PDE-304` when the equation is not of mKdV form or no real Miura map
/// exists.
pub fn miura(
    eq: ExprId,
    unknown: ExprId,
    kdv_unknown: ExprId,
    pool: &ExprPool,
) -> Result<Miura, PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    let (xi, ti) = time_space(&jets, pool)?;
    let p = 2;
    let vt = jets.jet(&alpha(p, ti, 1), pool);
    let vx = jets.jet(&alpha(p, xi, 1), pool);
    let vxxx = jets.jet(&alpha(p, xi, 3), pool);
    let not_mkdv = || {
        PdeError::TransformNotApplicable(format!(
            "{} is not of mKdV form v_t + a v^2 v_x + b v_xxx",
            pool.display(f)
        ))
    };
    let lam = d(f, vt, pool).ok_or_else(not_mkdv)?;
    if is_zero(lam, pool) || !is_constant(lam, &jets, pool) {
        return Err(not_mkdv());
    }
    let fl = simp(quotient(f, lam, pool), pool);
    let b = d(fl, vxxx, pool).ok_or_else(not_mkdv)?;
    let a2 = d(d(fl, vx, pool).ok_or_else(not_mkdv)?, jets.w, pool).ok_or_else(not_mkdv)?;
    let a = simp(
        quotient(
            d(a2, jets.w, pool).ok_or_else(not_mkdv)?,
            pool.integer(2),
            pool,
        ),
        pool,
    );
    let model = pool.add(vec![
        vt,
        pool.mul(vec![a, pool.pow(jets.w, pool.integer(2)), vx]),
        pool.mul(vec![b, vxxx]),
    ]);
    if is_zero(a, pool) || is_zero(b, pool) || !is_zero(minus(fl, model, pool), pool) {
        return Err(not_mkdv());
    }
    // u = α v² + β v_x; residual KdV[u] − (2αv + βD_x) mKdV[v] in v-jets.
    let al = pool.symbol("__miura_alpha", Domain::Real);
    let be = pool.symbol("__miura_beta", Domain::Real);
    let big_a = pool.symbol("__miura_A", Domain::Real);
    let u = pool.add(vec![
        pool.mul(vec![al, pool.pow(jets.w, pool.integer(2))]),
        pool.mul(vec![be, vx]),
    ]);
    let ut = jets.total_d(u, ti, None, pool);
    let ux = jets.total_d(u, xi, None, pool);
    let uxxx = jets.total_d_multi(u, &alpha(p, xi, 3), None, pool);
    let kdv = pool.add(vec![
        ut,
        pool.mul(vec![big_a, u, ux]),
        pool.mul(vec![b, uxxx]),
    ]);
    let op = pool.add(vec![
        pool.mul(vec![pool.integer(2), al, jets.w, model]),
        pool.mul(vec![be, jets.total_d(model, xi, None, pool)]),
    ]);
    let resid = crate::simplify::engine::simplify_expanded(minus(kdv, op, pool), pool).value;
    // Split by the jets.
    let gens: Vec<ExprId> = jets
        .jets_in(resid, pool)
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    let (rf, g) = crate::poly::cancel::rational_over_generators(resid, gens.clone(), pool)
        .map_err(|e| PdeError::TransformNotApplicable(e.to_string()))?;
    let ng = gens.len();
    let mut groups: HashMap<Vec<u32>, Vec<(Vec<u32>, rug::Integer)>> = HashMap::new();
    for (exps, c) in &rf.numer.terms {
        let get = |i: usize| exps.get(i).copied().unwrap_or(0);
        let key: Vec<u32> = (0..ng).map(get).collect();
        let mut rest: Vec<u32> = (0..g.len())
            .map(|i| if i < ng { 0 } else { get(i) })
            .collect();
        while rest.last() == Some(&0) {
            rest.pop();
        }
        groups.entry(key).or_default().push((rest, c.clone()));
    }
    let eqs: Vec<ExprId> = groups
        .into_values()
        .map(|ts| {
            crate::poly::multipoly::MultiPoly {
                vars: g.clone(),
                terms: ts.into_iter().collect(),
            }
            .to_expr(pool)
        })
        .collect();
    let sols = solve_small(&eqs, &[al, be, big_a], pool);
    if std::env::var("PDE_M6_DEBUG").is_ok() {
        eprintln!(
            "miura eqs {:?} -> {:?}",
            eqs.iter()
                .map(|&q| pool.display(q).to_string())
                .collect::<Vec<_>>(),
            sols.iter()
                .map(|s| format!(
                    "{} {} {}",
                    pool.display(s.0),
                    pool.display(s.1),
                    pool.display(s.2)
                ))
                .collect::<Vec<_>>()
        );
    }
    let Some((alv, bev, av)) = sols.into_iter().find(|(x, y, _)| {
        !is_zero(*x, pool)
            && !is_zero(*y, pool)
            && is_number(simp(pool.mul(vec![*x, *x]), pool), pool)
            && super::solve::util::eval_at(*x, &HashMap::new(), pool).is_some()
    }) else {
        return Err(PdeError::TransformNotApplicable(
            "no real Miura map u = α v² + β v_x exists for these coefficients".into(),
        ));
    };
    let m = HashMap::from([(al, alv), (be, bev), (big_a, av)]);
    let resid0 = simp(crate::kernel::subs::subs(resid, &m, pool), pool);
    let verified = is_zero(resid0, pool);
    let usub = simp(crate::kernel::subs::subs(u, &m, pool), pool);
    let kj = Jets::new(kdv_unknown, pool)?;
    let kdv_eq = simp(
        pool.add(vec![
            kj.jet(&alpha(p, ti, 1), pool),
            pool.mul(vec![av, kj.w, kj.jet(&alpha(p, xi, 1), pool)]),
            pool.mul(vec![b, kj.jet(&alpha(p, xi, 3), pool)]),
        ]),
        pool,
    );
    Ok(Miura {
        substitution: jets.space.from_jets(usub, pool),
        kdv: kj.space.from_jets(kdv_eq, pool),
        kdv_unknown,
        verified,
    })
}

/// Solve a small polynomial system in three unknowns: by Gröbner bases when
/// available, else by successive isolation.
fn solve_small(
    eqs: &[ExprId],
    vars: &[ExprId; 3],
    pool: &ExprPool,
) -> Vec<(ExprId, ExprId, ExprId)> {
    // The map is defined up to the scaling of u: fix α = ±1, solve for β, A.
    #[cfg(feature = "groebner")]
    {
        let mut out = Vec::new();
        for a in [1i64, -1] {
            let av = pool.integer(a);
            let m = HashMap::from([(vars[0], av)]);
            let mut sys: Vec<ExprId> = eqs
                .iter()
                .map(|&e| simp(crate::kernel::subs::subs(e, &m, pool), pool))
                .collect();
            let bi = pool.symbol("__miura_bi", Domain::Real);
            sys.push(minus(pool.mul(vec![vars[1], bi]), pool.integer(1), pool));
            let res = crate::solver::solve_polynomial_system(sys, vec![vars[1], vars[2], bi], pool);
            let _ = crate::solver::take_solve_side_conditions();
            if let Ok(crate::solver::SolutionSet::Finite(sols)) = res {
                out.extend(sols.into_iter().map(|s| (av, s[0], s[1])));
            }
        }
        out
    }
    #[cfg(not(feature = "groebner"))]
    {
        let _ = (eqs, vars, pool);
        Vec::new()
    }
}

/// The hodograph transformation of a 2×2 quasilinear system.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Hodograph {
    /// The linear system for `x(u, v)`, `t(u, v)` (`= 0`).
    pub equations: Vec<ExprId>,
    /// `x(u, v)` and `t(u, v)` as applied functions.
    pub new_unknowns: Vec<ExprId>,
    /// The Jacobian `x_u t_v − x_v t_u`, assumed non-zero.
    pub jacobian: ExprId,
    /// The transformed equations equal `J ·` the original ones exactly.
    pub verified: bool,
}

/// The hodograph transformation of `Aᵢ u_t + Bᵢ u_x + Cᵢ v_t + Dᵢ v_x = 0`
/// (`i = 1, 2`, coefficients functions of `u, v` only): exchanging the roles
/// of `(x, t)` and `(u, v)` gives the **linear** system
/// `−Aᵢ x_v + Bᵢ t_v + Cᵢ x_u − Dᵢ t_u = 0` for `x(u, v)`, `t(u, v)`, valid
/// where the Jacobian does not vanish.
///
/// # Errors
///
/// `E-PDE-304` for a system that is not of that form; M0 errors for
/// malformed unknowns.
pub fn hodograph(
    equations: &[ExprId],
    unknowns: &[ExprId],
    pool: &ExprPool,
) -> Result<Hodograph, PdeError> {
    if equations.len() != 2 || unknowns.len() != 2 {
        return Err(PdeError::TransformNotApplicable(
            "the hodograph transformation takes two equations in two unknowns".into(),
        ));
    }
    let v0 = super::jet::decode_unknown(unknowns[0], pool)?;
    let indep = v0.args;
    if indep.len() != 2 {
        return Err(PdeError::TransformNotApplicable(
            "the hodograph transformation needs two independent variables".into(),
        ));
    }
    let space = JetSpace::new(&indep, unknowns, 1, pool)?;
    let ti = indep
        .iter()
        .position(|&x| pool.display(x).to_string() == "t")
        .unwrap_or(1);
    let xi = 1 - ti;
    let u = space.jet(0, &[0, 0], pool)?;
    let v = space.jet(1, &[0, 0], pool)?;
    let ut = space.jet(0, &alpha(2, ti, 1), pool)?;
    let ux = space.jet(0, &alpha(2, xi, 1), pool)?;
    let vt = space.jet(1, &alpha(2, ti, 1), pool)?;
    let vx = space.jet(1, &alpha(2, xi, 1), pool)?;
    let (x, t) = (indep[xi], indep[ti]);
    let not_quasi = |e: ExprId| {
        PdeError::TransformNotApplicable(format!(
            "{} is not homogeneous quasilinear first order with coefficients in (u, v)",
            pool.display(e)
        ))
    };
    // New unknowns x(u, v), t(u, v) with names avoiding the old ones.
    let (us, vs) = (
        pool.symbol(pool.display(u).to_string(), Domain::Real),
        pool.symbol(pool.display(v).to_string(), Domain::Real),
    );
    let xn = pool.func(pool.display(x).to_string() + "_h", vec![us, vs]);
    let tn = pool.func(pool.display(t).to_string() + "_h", vec![us, vs]);
    let dd = |f: ExprId, a: u32, b: u32| crate::diff::undefined::build(f, &[a, b], pool);
    let (xu, xv, tu, tv) = (dd(xn, 1, 0), dd(xn, 0, 1), dd(tn, 1, 0), dd(tn, 0, 1));
    let jac = minus(pool.mul(vec![xu, tv]), pool.mul(vec![xv, tu]), pool);
    let mut out = Vec::new();
    let mut verified = true;
    for &e in equations {
        let f = simp(space.to_jets(e, pool)?, pool);
        let mut cs = Vec::new();
        for s in [ut, ux, vt, vx] {
            let c = d(f, s, pool).ok_or_else(|| not_quasi(e))?;
            if contains(c, x, pool)
                || contains(c, t, pool)
                || [ut, ux, vt, vx].iter().any(|&j| contains(c, j, pool))
            {
                return Err(not_quasi(e));
            }
            cs.push(c);
        }
        let lin = pool.add(vec![
            pool.mul(vec![cs[0], ut]),
            pool.mul(vec![cs[1], ux]),
            pool.mul(vec![cs[2], vt]),
            pool.mul(vec![cs[3], vx]),
        ]);
        if !is_zero(minus(f, lin, pool), pool) {
            return Err(not_quasi(e));
        }
        let m = HashMap::from([(u, us), (v, vs)]);
        let cs: Vec<ExprId> = cs
            .iter()
            .map(|&c| simp(crate::kernel::subs::subs(c, &m, pool), pool))
            .collect();
        let l = simp(
            pool.add(vec![
                pool.mul(vec![pool.integer(-1), cs[0], xv]),
                pool.mul(vec![cs[1], tv]),
                pool.mul(vec![cs[2], xu]),
                pool.mul(vec![pool.integer(-1), cs[3], tu]),
            ]),
            pool,
        );
        // Identity check: J·(A u_t + B u_x + C v_t + D v_x) with the inverse
        // Jacobian relations equals the linear form.
        let inv = |num: ExprId| quotient(num, jac, pool);
        let back = pool.add(vec![
            pool.mul(vec![cs[0], inv(pool.mul(vec![pool.integer(-1), xv]))]),
            pool.mul(vec![cs[1], inv(tv)]),
            pool.mul(vec![cs[2], inv(xu)]),
            pool.mul(vec![cs[3], inv(pool.mul(vec![pool.integer(-1), tu]))]),
        ]);
        if !is_zero(minus(pool.mul(vec![back, jac]), l, pool), pool) {
            verified = false;
        }
        out.push(l);
    }
    Ok(Hodograph {
        equations: out,
        new_unknowns: vec![xn, tn],
        jacobian: simp(jac, pool),
        verified,
    })
}

/// The Hopf–Lax data of `u_t + H(u_x) = 0`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct HopfLax {
    /// `H(p)` in the symbol `p`.
    pub hamiltonian: ExprId,
    /// The symbol `p`.
    pub p: ExprId,
    /// The Lagrangian `L(q) = sup_p (qp − H(p))` in the symbol `q`.
    pub lagrangian: ExprId,
    /// The symbol `q`.
    pub q: ExprId,
    /// The objective `g(y) + t L((x − y)/t)` minimised over `y`
    /// (`u(x, t) = min_y …` for data `u(x, 0) = g(x)`).
    pub objective: ExprId,
    /// The minimisation variable `y`.
    pub y: ExprId,
    /// `L'(H'(p)) = p` proved exactly.
    pub verified: bool,
    /// The convexity hypothesis `H''(p) > 0` (not decided symbolically).
    pub convexity: ExprId,
}

/// The Hopf–Lax formula for `u_t + H(u_x) = 0`: the Legendre transform `L`
/// of `H` (from `q = H'(p)` solved for `p`) and the objective
/// `g(y) + t L((x − y)/t)`; valid for convex, superlinear `H`.
///
/// # Errors
///
/// `E-PDE-304` when the equation is not `u_t + H(u_x)` or `q = H'(p)` cannot
/// be solved for `p`.
pub fn hopf_lax(eq: ExprId, unknown: ExprId, pool: &ExprPool) -> Result<HopfLax, PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    let (xi, ti) = time_space(&jets, pool)?;
    let ut = jets.jet(&alpha(2, ti, 1), pool);
    let ux = jets.jet(&alpha(2, xi, 1), pool);
    let not_hj = || {
        PdeError::TransformNotApplicable(format!(
            "{} is not of Hamilton–Jacobi form u_t + H(u_x)",
            pool.display(f)
        ))
    };
    let lam = d(f, ut, pool).ok_or_else(not_hj)?;
    if is_zero(lam, pool) || !is_constant(lam, &jets, pool) {
        return Err(not_hj());
    }
    let h = simp(minus(quotient(f, lam, pool), ut, pool), pool);
    let others: Vec<ExprId> = jets
        .jets_in(h, pool)
        .into_iter()
        .map(|(_, s)| s)
        .filter(|&s| s != ux)
        .collect();
    if !others.is_empty()
        || jets.indep.iter().any(|&x| contains(h, x, pool))
        || !contains(h, ux, pool)
    {
        return Err(not_hj());
    }
    let mut taken = super::solve::util::names_in(&[eq], pool);
    let mut sym = |s: &str| {
        if taken.insert(s.to_string()) {
            pool.symbol(s, Domain::Real)
        } else {
            super::solve::util::fresh_symbol(s, &mut taken, pool)
        }
    };
    let (p, q, y) = (sym("p"), sym("q"), sym("y"));
    let hp = simp(
        crate::kernel::subs::subs(h, &HashMap::from([(ux, p)]), pool),
        pool,
    );
    let dh = d(hp, p, pool).ok_or_else(not_hj)?;
    let pq = isolate(minus(dh, q, pool), p, pool)
        .into_iter()
        .find(|&v| !contains(v, p, pool))
        .ok_or_else(|| {
            PdeError::TransformNotApplicable(format!(
                "q = H'(p) = {} cannot be solved for p",
                pool.display(dh)
            ))
        })?;
    let l = simp(
        minus(
            pool.mul(vec![q, pq]),
            crate::kernel::subs::subs(hp, &HashMap::from([(p, pq)]), pool),
            pool,
        ),
        pool,
    );
    // Legendre duality: L'(q) = p(q).
    let verified = d(l, q, pool).is_some_and(|lq| is_zero(minus(lq, pq, pool), pool));
    let (x, t) = (jets.indep[xi], jets.indep[ti]);
    let g = pool.func("g", vec![y]);
    let arg = quotient(minus(x, y, pool), t, pool);
    let objective = pool.add(vec![
        g,
        pool.mul(vec![
            t,
            crate::kernel::subs::subs(l, &HashMap::from([(q, arg)]), pool),
        ]),
    ]);
    let convexity = d(dh, p, pool).unwrap_or_else(|| pool.integer(0));
    Ok(HopfLax {
        hamiltonian: hp,
        p,
        lagrangian: l,
        q,
        objective: simp(objective, pool),
        y,
        verified,
        convexity,
    })
}
