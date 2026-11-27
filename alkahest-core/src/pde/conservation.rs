//! Conservation laws of a scalar PDE by the multiplier method (milestone M6)
//! — the equivalent of Maple PDEtools' `ConservedCurrents`, GeM and CONLAW.
//!
//! * [`euler_operator`] — the variational derivative
//!   `E_u(L) = Σ_J (−D)_J ∂L/∂u_J` on M0's jet space.
//! * [`conservation_multipliers`] — multipliers `Λ(x, u, ∂u, …)` (of a chosen
//!   order, depending only on derivatives that are not derivatives of the
//!   leading one, so that trivial laws are excluded) with
//!   `E_u(Λ·F) ≡ 0`.  That condition is a linear overdetermined system for
//!   `Λ`, split by the jets `Λ` does not depend on and solved by the same
//!   RIF-style engine as the symmetry determining systems.
//! * [`conserved_flux`] — the density/flux pair `Σ_i D_{x_i} Φ^i = Λ·F`,
//!   recovered by successive integration of the highest derivative (a
//!   homotopy-free inversion of the total divergence) and **verified
//!   exactly** with M0's total derivatives.
//!
//! Expressions are returned in jet notation (`u`, `u_x`, `u_xx`, …), the
//! symbols of [`super::JetSpace`]; `JetSpace.from_jets` converts to
//! derivative nodes.
//!
//! Errors: `E-PDE-300`/`301` as for symmetries, `E-PDE-305` (unsupported),
//! `E-PDE-306` (no flux found).

use std::collections::HashMap;

use crate::kernel::{ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::detsys::{atom_sym, Alpha, DetSys, Lin};
use super::error::PdeError;
use super::solve::util::{contains, d, is_zero, simp};
use super::symmetry::{parameters_of, rank_key, split_by_jets, Jets, Reducer};

/// A conservation law `Σ_i D_{x_i} Φ^i = Λ·F`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConservationLaw {
    /// The multiplier `Λ` (jet notation).
    pub multiplier: ExprId,
    /// `Φ^i`, one per independent variable (jet notation); empty when the
    /// flux was not recovered.
    pub components: Vec<ExprId>,
    /// Whether `Σ D_i Φ^i − Λ·F = 0` was proved exactly.
    pub verified: bool,
}

/// The multipliers of one equation up to a given order.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Multipliers {
    /// Independent variables.
    pub independent: Vec<ExprId>,
    /// The jet coordinates the multipliers may depend on.
    pub coordinates: Vec<ExprId>,
    /// A basis of the finite part.
    pub multipliers: Vec<ExprId>,
    /// Multipliers involving arbitrary functions (`F1(x - t)` for the wave
    /// equation), with the equations those functions satisfy.
    pub infinite: Option<(ExprId, Vec<ExprId>)>,
    /// Every equation of the multiplier system was resolved.
    pub complete: bool,
    /// Parameter case (`param = value`) of this answer.
    pub case: Vec<(ExprId, ExprId)>,
    /// Notes.
    pub notes: Vec<String>,
}

fn setup(eq: ExprId, unknown: ExprId, pool: &ExprPool) -> Result<(Jets, ExprId), PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let f = simp(jets.to_jets(eq, pool)?, pool);
    if jets
        .jets_in(f, pool)
        .iter()
        .all(|(a, _)| a.iter().sum::<u32>() == 0)
    {
        return Err(PdeError::NotADifferentialEquation(format!(
            "{} has no derivative of {}",
            pool.display(eq),
            pool.display(unknown)
        )));
    }
    Ok((jets, f))
}

/// `E_u(L) = Σ_J (−D)_J ∂L/∂u_J` in jets (atoms of `sys` allowed).
fn euler_jets(l: ExprId, jets: &Jets, sys: Option<&DetSys>, pool: &ExprPool) -> ExprId {
    let mut terms = Vec::new();
    let mut syms: Vec<(Alpha, ExprId)> = jets.jets_in(l, pool);
    // Jets reachable through atoms (Λ's coordinates).
    if let Some(s) = sys {
        for &z in &s.vars {
            if let Some(a) = jets.decode(z, pool) {
                if !syms.iter().any(|(_, y)| *y == z) {
                    syms.push((a, z));
                }
            }
        }
    }
    for (a, s) in syms {
        let Some(mut part) = partial_with_atoms(l, s, sys, pool) else {
            continue;
        };
        if part == pool.integer(0) {
            continue;
        }
        for (i, &k) in a.iter().enumerate() {
            for _ in 0..k {
                part = jets.total_d(part, i, sys, pool);
                part = simp(pool.mul(vec![pool.integer(-1), part]), pool);
            }
        }
        terms.push(part);
    }
    if terms.is_empty() {
        pool.integer(0)
    } else {
        simp(pool.add(terms), pool)
    }
}

/// `∂L/∂z` where `L` may contain atoms of functions depending on `z`.
fn partial_with_atoms(
    l: ExprId,
    z: ExprId,
    sys: Option<&DetSys>,
    pool: &ExprPool,
) -> Option<ExprId> {
    let mut terms = vec![d(l, z, pool)?];
    if let Some(sys) = sys {
        let j = sys.vars.iter().position(|&v| v == z);
        if let Some(j) = j {
            for s in super::jet::symbols_of(l, pool) {
                let Some(a) = super::detsys::parse_atom_sym(s, pool) else {
                    continue;
                };
                if !sys.funcs[a.0].deps.contains(&j) {
                    continue;
                }
                let pd = d(l, s, pool)?;
                if pd == pool.integer(0) {
                    continue;
                }
                let mut b = a.1.clone();
                b[j] += 1;
                terms.push(pool.mul(vec![pd, atom_sym(&(a.0, b), pool)]));
            }
        }
    }
    Some(simp(pool.add(terms), pool))
}

/// The Euler operator (variational derivative) `E_u(L)` of a Lagrangian
/// density `L` (written with derivative nodes or jets), in jet notation.
///
/// # Errors
///
/// M0 errors for a malformed unknown or an expression off the jet space.
pub fn euler_operator(
    lagrangian: ExprId,
    unknown: ExprId,
    pool: &ExprPool,
) -> Result<ExprId, PdeError> {
    let jets = Jets::new(unknown, pool)?;
    let l = simp(jets.to_jets(lagrangian, pool)?, pool);
    let e = euler_jets(l, &jets, None, pool);
    Ok(crate::simplify::engine::simplify_expanded(e, pool).value)
}

/// The jets a multiplier of order `order` may depend on: derivatives up to
/// `order` that are not derivatives of the leading one.
fn coordinates(jets: &Jets, red: &Reducer, order: u32, pool: &ExprPool) -> Vec<ExprId> {
    let p = jets.p();
    let mut out = jets.indep.clone();
    out.push(jets.w);
    for k in 1..=order {
        let mut ms = super::jet::multi_indices(p, k);
        ms.sort_by_key(|a| rank_key(a));
        for a in ms {
            if a.iter().zip(&red.lead).all(|(x, y)| x >= y) {
                continue;
            }
            out.push(jets.jet(&a, pool));
        }
    }
    out
}

/// Multipliers `Λ` of `eq = 0` depending on the independent variables, `u`
/// and the non-leading derivatives up to `order`, one answer per parameter
/// case.
///
/// # Errors
///
/// `E-PDE-300` / `E-PDE-301` as for [`super::point_symmetries`];
/// `E-PDE-305` for an order above 6.
pub fn conservation_multipliers(
    eq: ExprId,
    unknown: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Vec<Multipliers>, PdeError> {
    if order > 6 {
        return Err(PdeError::ConservationUnsupported(format!(
            "multiplier order {order} is above the supported 6"
        )));
    }
    let (jets, f) = setup(eq, unknown, pool)?;
    let red = Reducer::new(&jets, f, pool)?;
    let vars = coordinates(&jets, &red, order, pool);
    let params = parameters_of(f, &jets, pool);
    let mut sys = DetSys::new(vars.clone(), params);
    let all: Vec<usize> = (0..vars.len()).collect();
    sys.add_func("Lambda", all);
    let za = sys.zero_alpha();
    let mut t = Lin::new();
    t.insert((0, za.clone()), pool.integer(1));
    sys.targets.push(t);
    let lam = atom_sym(&(0, za), pool);
    let h = simp(pool.mul(vec![lam, f]), pool);
    let e = euler_jets(h, &jets, Some(&sys), pool);
    let split: Vec<ExprId> = jets
        .jets_in(e, pool)
        .into_iter()
        .map(|(_, s)| s)
        .filter(|s| !vars.contains(s))
        .collect();
    let eqs = split_by_jets(e, &split, &vars, &jets, pool)?;
    for q in eqs {
        sys.push_eq(q);
    }
    let mut out = Vec::new();
    for b in sys.solve(pool) {
        let live = b.live_funcs();
        let constrained = b.constrained();
        let mut finite = Vec::new();
        let mut inf_funcs = Vec::new();
        for &k in &live {
            if b.funcs[k].deps.is_empty() && !constrained.contains(&k) {
                finite.push(b.targets_for_constant(k, pool)[0]);
            } else {
                inf_funcs.push(k);
            }
        }
        // Basis of the finite part.
        let mut basis: Vec<ExprId> = Vec::new();
        for m in finite {
            let m = simp(m, pool);
            if is_zero(m, pool) {
                continue;
            }
            let rows: Vec<Vec<ExprId>> = basis.iter().map(|&x| vec![x]).collect();
            if !rows.is_empty() && super::symmetry::decompose(&[m], &rows, &vars, pool).is_some() {
                continue;
            }
            basis.push(m);
        }
        basis.sort_by_key(|&m| super::jet::symbols_of(m, pool).len());
        let mut complete = b.complete;
        let infinite = if inf_funcs.is_empty() && b.eqs.is_empty() {
            None
        } else {
            let mut names = HashMap::new();
            let (mut nf, mut nc) = (0, 0);
            for &k in &inf_funcs {
                let n = if b.funcs[k].deps.is_empty() {
                    nc += 1;
                    format!("C{nc}")
                } else {
                    nf += 1;
                    format!("F{nf}")
                };
                names.insert(k, n);
            }
            let l: Lin = b.targets[0]
                .iter()
                .filter(|(a, _)| inf_funcs.contains(&a.0))
                .map(|(a, &c)| (a.clone(), c))
                .collect();
            let m = b.lin_to_expr(&l, &names, pool);
            let cons = b
                .eqs
                .iter()
                .map(|q| b.lin_to_expr(q, &names, pool))
                .collect();
            if !b.eqs.is_empty() && !b.passive {
                complete = false;
            }
            Some((m, cons))
        };
        out.push(Multipliers {
            independent: jets.indep.clone(),
            coordinates: vars.clone(),
            multipliers: basis,
            infinite,
            complete,
            case: b.case.clone(),
            notes: b.notes.clone(),
        });
    }
    out.sort_by_key(|m| m.case.len());
    Ok(out)
}

/// Preferred direction for a flux integration: a variable named `t`, else
/// the last one.
fn time_index(jets: &Jets, pool: &ExprPool) -> usize {
    jets.indep
        .iter()
        .position(|&x| pool.display(x).to_string() == "t")
        .unwrap_or(jets.p() - 1)
}

/// The density/flux of the conservation law with multiplier `multiplier`
/// (jet notation): `Φ` with `Σ_i D_{x_i} Φ^i = Λ·F` identically, verified
/// exactly.
///
/// # Errors
///
/// `E-PDE-306` when `Λ·F` could not be inverted as a divergence (in
/// particular when `Λ` is not a multiplier).
pub fn conserved_flux(
    eq: ExprId,
    unknown: ExprId,
    multiplier: ExprId,
    pool: &ExprPool,
) -> Result<ConservationLaw, PdeError> {
    let (jets, f) = setup(eq, unknown, pool)?;
    let lam = simp(jets.to_jets(multiplier, pool)?, pool);
    let target = crate::simplify::engine::simplify_expanded(pool.mul(vec![lam, f]), pool).value;
    let comps = invert_divergence(target, &jets, pool).ok_or_else(|| {
        PdeError::FluxNotRecovered(format!(
            "{} · ({}) was not inverted as a total divergence",
            pool.display(lam),
            pool.display(f)
        ))
    })?;
    let st = verify_divergence(&jets, &comps, target, pool);
    if st != ZeroStatus::ProvedZero {
        return Err(PdeError::FluxNotRecovered(format!(
            "the recovered flux did not verify ({})",
            st.as_str()
        )));
    }
    Ok(ConservationLaw {
        multiplier: lam,
        components: comps,
        verified: true,
    })
}

/// `Σ D_i Φ^i − target`, zero-tested (M0 total derivatives).
fn verify_divergence(jets: &Jets, comps: &[ExprId], target: ExprId, pool: &ExprPool) -> ZeroStatus {
    let mut terms = vec![pool.mul(vec![pool.integer(-1), target])];
    for (i, &c) in comps.iter().enumerate() {
        match jets.space.total_derivative(c, i, pool) {
            Ok(dc) => terms.push(dc),
            Err(_) => return ZeroStatus::Unknown,
        }
    }
    let r = crate::simplify::engine::simplify_expanded(pool.add(terms), pool).value;
    if r == pool.integer(0) {
        return ZeroStatus::ProvedZero;
    }
    zero_test(r, pool, None).status
}

/// Successive integration of the highest-ranked derivative.
fn invert_divergence(target: ExprId, jets: &Jets, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let p = jets.p();
    let ti = time_index(jets, pool);
    let mut comps: Vec<Vec<ExprId>> = vec![Vec::new(); p];
    let mut r = target;
    for _ in 0..80 {
        if crate::budget::tripped().is_some() {
            return None;
        }
        if is_zero(r, pool) {
            let out = comps
                .into_iter()
                .map(|v| {
                    if v.is_empty() {
                        pool.integer(0)
                    } else {
                        crate::simplify::engine::simplify_expanded(pool.add(v), pool).value
                    }
                })
                .collect();
            return Some(out);
        }
        let js: Vec<(Alpha, ExprId)> = jets
            .jets_in(r, pool)
            .into_iter()
            .filter(|(a, _)| a.iter().sum::<u32>() > 0)
            .collect();
        let Some((k, sym)) = js
            .into_iter()
            .max_by(|a, b| rank_key(&a.0).cmp(&rank_key(&b.0)))
        else {
            // No derivatives: only a function of the independent variables
            // is a divergence here.
            if contains(r, jets.w, pool) {
                return None;
            }
            let x = jets.indep[0];
            let i = super::solve::util::integrate(r, x, pool)?;
            comps[0].push(i);
            r = simp(
                pool.add(vec![r, pool.mul(vec![pool.integer(-1), d(i, x, pool)?])]),
                pool,
            );
            continue;
        };
        let a = d(r, sym, pool)?;
        if contains(a, sym, pool) {
            return None;
        }
        // Direction: time first when the derivative has a time component.
        let i = if k[ti] > 0 {
            ti
        } else {
            (0..p).find(|&i| k[i] > 0)?
        };
        let mut lower = k.clone();
        lower[i] -= 1;
        let w = jets.jet(&lower, pool);
        let pint = super::solve::util::integrate(a, w, pool)?;
        // Exactness of the antiderivative in w.
        let back = d(pint, w, pool)?;
        if !is_zero(
            pool.add(vec![back, pool.mul(vec![pool.integer(-1), a])]),
            pool,
        ) {
            return None;
        }
        let dp = jets.total_d(pint, i, None, pool);
        comps[i].push(pint);
        r = crate::simplify::engine::simplify_expanded(
            pool.add(vec![r, pool.mul(vec![pool.integer(-1), dp])]),
            pool,
        )
        .value;
    }
    None
}

/// Every conservation law with a multiplier of order `≤ order`: the
/// multipliers of [`conservation_multipliers`] (generic case) with their
/// verified density/flux; a multiplier whose flux is not recovered is kept
/// with empty `components` and `verified = false`.
///
/// # Errors
///
/// As [`conservation_multipliers`].
pub fn conservation_laws(
    eq: ExprId,
    unknown: ExprId,
    order: u32,
    pool: &ExprPool,
) -> Result<Vec<ConservationLaw>, PdeError> {
    let ms = conservation_multipliers(eq, unknown, order, pool)?;
    let Some(m) = ms.into_iter().next() else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for lam in m.multipliers {
        match conserved_flux(eq, unknown, lam, pool) {
            Ok(c) => out.push(c),
            Err(_) => out.push(ConservationLaw {
                multiplier: lam,
                components: Vec::new(),
                verified: false,
            }),
        }
    }
    Ok(out)
}

/// Whether `Σ_i D_{x_i} Φ^i = Λ·F` holds identically (components and
/// multiplier in jets or derivative nodes).
///
/// # Errors
///
/// M0 errors; `E-PDE-013` for the wrong number of components.
pub fn verify_conservation_law(
    eq: ExprId,
    unknown: ExprId,
    multiplier: ExprId,
    components: &[ExprId],
    pool: &ExprPool,
) -> Result<ZeroStatus, PdeError> {
    let (jets, f) = setup(eq, unknown, pool)?;
    if components.len() != jets.p() {
        return Err(PdeError::InvalidCandidate(format!(
            "a conservation law needs {} components, got {}",
            jets.p(),
            components.len()
        )));
    }
    let lam = jets.to_jets(multiplier, pool)?;
    let comps: Vec<ExprId> = components
        .iter()
        .map(|&c| jets.to_jets(c, pool))
        .collect::<Result<_, _>>()?;
    let target = simp(pool.mul(vec![lam, f]), pool);
    Ok(verify_divergence(&jets, &comps, target, pool))
}
