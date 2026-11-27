//! The ordinary differential equation each mode's amplitude satisfies in the
//! non-eigen direction `s` (time, or the direction of a boundary-value
//! problem that carries the data):
//!
//! ```text
//! a₂ T'' + a₁ T' + a₀ T = q(s)
//! ```
//!
//! with initial values (an evolution problem) or two-point conditions
//! (`αT + βT' = b` at each end, or boundedness at a singular end `s = 0`).
//!
//! Recognised forms, after dividing by `a₂`:
//! * constant coefficients — `e^{−σz}(C(z), S(z))` with `C = cos ωz`,
//!   `S = sin(ωz)/ω` (or `cosh`, `sinh(κz)/κ`, or `1`, `z` when `ω = 0`).
//!   Both are even in `ω`, so the formulas are correct for every value of
//!   `ω² = a₀/a₂ − σ²` — the choice of trigonometric or hyperbolic form is
//!   cosmetic;
//! * Euler `T'' + (β/s)T' + (γ/s²)T` — `s^{p±}` (or `s^p`, `s^p log s`);
//! * Bessel `T'' + T'/s + (A + B/s²)T` — `J_ν(ωs)`, `Y_ν(ωs)` or
//!   `I_ν(κs)`, `K_ν(κs)`;
//! * spherical Bessel `T'' + 2T'/s + (A + B/s²)T` — `s^{−1/2}` times those
//!   with `ν² = 1/4 − B`.
//!
//! The source term goes through Duhamel's integral, closed by `integrate`
//! (antiderivative checked exactly) or left as an unevaluated `Integral`.

use std::collections::HashMap;

use crate::kernel::subs::{mentions_var, subs};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::simplify::zero::{zero_test, ZeroStatus};

use super::super::error::PdeError;
use super::super::sturm_liouville::{d, sign, simp, subs1, Sgn};

pub(crate) fn neg(e: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![pool.integer(-1), e])
}
pub(crate) fn sub(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.add(vec![a, neg(b, pool)])
}
pub(crate) fn div(a: ExprId, b: ExprId, pool: &ExprPool) -> ExprId {
    pool.mul(vec![a, pool.pow(b, pool.integer(-1))])
}
pub(crate) fn is_zero_expr(e: ExprId, pool: &ExprPool) -> bool {
    let s = simp(e, pool);
    s == pool.integer(0) || zero_test(s, pool, None).status == ZeroStatus::ProvedZero
}

/// A square root chosen structurally: even powers and perfect-square
/// rationals are taken out (`√((nπ/L)²) = nπ/L`).  Correctness never depends
/// on the branch here: every formula that uses it is even in the root.
pub(crate) fn sqrt_pos(e: ExprId, pool: &ExprPool) -> ExprId {
    let e = simp(e, pool);
    let half = pool.rational(1, 2);
    let one = |f: ExprId| -> Option<ExprId> {
        match pool.get(f) {
            ExprData::Pow { base, exp } => {
                let k = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.to_i64(),
                    _ => None,
                })?;
                (k % 2 == 0).then(|| pool.pow(base, pool.integer(k / 2)))
            }
            ExprData::Integer(n) => {
                let v = n.0.to_i64()?;
                if v < 0 {
                    return None;
                }
                let r = (v as f64).sqrt().round() as i64;
                (r * r == v).then(|| pool.integer(r))
            }
            ExprData::Rational(q) => {
                let (n, dd) = (q.0.numer().to_i64()?, q.0.denom().to_i64()?);
                if n < 0 {
                    return None;
                }
                let (rn, rd) = (
                    (n as f64).sqrt().round() as i64,
                    (dd as f64).sqrt().round() as i64,
                );
                (rn * rn == n && rd * rd == dd).then(|| pool.rational(rn, rd))
            }
            _ => None,
        }
    };
    let factors = match pool.get(e) {
        ExprData::Mul(xs) => xs,
        _ => vec![e],
    };
    let mut out = Vec::new();
    let mut rest = Vec::new();
    for f in factors {
        match one(f) {
            Some(r) => out.push(r),
            None => rest.push(f),
        }
    }
    if !rest.is_empty() {
        out.push(pool.pow(pool.mul(rest), half));
    }
    simp(pool.mul(out), pool)
}

/// Sign used only to choose a *cosmetic* form: symbols in `positive` are
/// treated as positive, as are even powers.
pub(crate) fn sign_hint(e: ExprId, positive: &[ExprId], pool: &ExprPool) -> Sgn {
    // Positive roots (branch 0) of a RootOfFamily are positive.
    let e = {
        let mut rm = HashMap::new();
        let mut stack = vec![e];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            if let Some((_, _, _, 0)) = super::super::root_family::root_family_parts(n, pool) {
                rm.insert(
                    n,
                    pool.symbol(format!("__rof_pos_{}", n.0), Domain::Positive),
                );
                continue;
            }
            super::super::jet::push_children(n, pool, &mut stack);
        }
        if rm.is_empty() {
            e
        } else {
            super::separate::replace_nodes(e, &rm, pool)
        }
    };
    let mut m = HashMap::new();
    for &s in positive {
        if let ExprData::Symbol { name, .. } = pool.get(s) {
            m.insert(s, pool.symbol(format!("{name}__pos"), Domain::Positive));
        }
    }
    let e2 = simp(subs(e, &m, pool), pool);
    let s = sign(e2, pool);
    if s != Sgn::Unknown {
        return s;
    }
    // Even powers of real symbols: squares.
    fn even_sign(e: ExprId, pool: &ExprPool) -> Sgn {
        match pool.get(e) {
            ExprData::Pow { base: _, exp } => {
                let even = pool.with(exp, |d| match d {
                    ExprData::Integer(n) => n.0.is_even(),
                    _ => false,
                });
                if even {
                    Sgn::Pos
                } else {
                    Sgn::Unknown
                }
            }
            ExprData::Mul(xs) => {
                let mut negs = 0;
                for x in xs {
                    match even_sign(x, pool) {
                        Sgn::Pos => {}
                        Sgn::Neg => negs += 1,
                        _ => {
                            let s = sign(x, pool);
                            match s {
                                Sgn::Pos => {}
                                Sgn::Neg => negs += 1,
                                _ => return Sgn::Unknown,
                            }
                        }
                    }
                }
                if negs % 2 == 0 {
                    Sgn::Pos
                } else {
                    Sgn::Neg
                }
            }
            ExprData::Add(xs) => {
                let ss: Vec<Sgn> = xs.iter().map(|&x| even_sign(x, pool)).collect();
                if ss.iter().all(|s| *s == Sgn::Pos) {
                    Sgn::Pos
                } else if ss.iter().all(|s| *s == Sgn::Neg) {
                    Sgn::Neg
                } else {
                    Sgn::Unknown
                }
            }
            _ => sign(e, pool),
        }
    }
    even_sign(e2, pool)
}

/// A condition at one end of the `s` interval.
#[derive(Clone, Debug)]
pub(crate) enum SEnd {
    /// `α T(at) + β T'(at) = data`.
    Face {
        at: ExprId,
        alpha: ExprId,
        beta: ExprId,
        data: ExprId,
    },
    /// `T` bounded at a singular end (`s = 0`).
    Bounded,
}

/// The data of the mode problem.
#[derive(Clone, Debug)]
pub(crate) enum SData {
    /// `T(t0) = v[0]`, `T'(t0) = v[1]`.
    Initial { t0: ExprId, values: Vec<ExprId> },
    /// Conditions at `s0` and `s1`.
    TwoPoint {
        s0: ExprId,
        s1: ExprId,
        left: SEnd,
        right: SEnd,
    },
}

/// The solution `T(s)` of one mode problem.
#[derive(Clone, Debug)]
pub(crate) struct ModeSolution {
    pub value: ExprId,
    /// Whether every part is closed (no unevaluated Duhamel integral).
    pub closed: bool,
    pub side_conditions: Vec<ExprId>,
    pub notes: Vec<String>,
    /// Fresh arbitrary constants introduced (a Neumann problem's constant).
    pub constants: Vec<ExprId>,
    /// `"constant_coefficients"`, `"euler"`, `"bessel"`, `"spherical_bessel"`.
    pub form: &'static str,
}

/// Normalised coefficients `T'' + p T' + q T` (order 2) or `T' + q T`
/// (order 1, `p` unused).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModeOde {
    pub order: u32,
    pub a2: ExprId,
    pub a1: ExprId,
    pub a0: ExprId,
}

/// Read `a₂, a₁, a₀` off a linear homogeneous ODE in `factor(s)`.
pub(crate) fn mode_ode(ode: ExprId, factor: ExprId, pool: &ExprPool) -> Result<ModeOde, PdeError> {
    use crate::diff::undefined::build;
    let mut e = ode;
    let mut syms = Vec::new();
    for k in (0..3u32).rev() {
        let node = build(factor, &[k], pool);
        let s = pool.symbol(format!("__mode_y{k}"), Domain::Real);
        e = super::separate::replace_nodes(e, &HashMap::from([(node, s)]), pool);
        syms.push(s);
    }
    syms.reverse();
    let name = crate::diff::undefined::view(factor, pool)
        .map(|v| v.name)
        .unwrap_or_default();
    if super::super::problem::mentions_unknown(e, &[name], pool) {
        return Err(PdeError::SeparationModeOde(format!(
            "{} has derivatives above the second in {}",
            pool.display(ode),
            pool.display(factor)
        )));
    }
    let mut c = Vec::new();
    for &y in &syms {
        let cy = simp(d(e, y, pool)?, pool);
        if syms.iter().any(|&z| mentions_var(cy, z, pool)) {
            return Err(PdeError::SeparationModeOde(format!(
                "{} is not linear",
                pool.display(ode)
            )));
        }
        c.push(cy);
    }
    let zero = pool.integer(0);
    let order = if c[2] != zero {
        2
    } else if c[1] != zero {
        1
    } else {
        0
    };
    Ok(ModeOde {
        order,
        a2: c[2],
        a1: c[1],
        a0: c[0],
    })
}

/// `C(z)` and `S(z)` for `ω² = w2` (cosmetic choice of form by `sign_hint`).
fn cs(z: ExprId, w2: ExprId, positive: &[ExprId], pool: &ExprPool) -> (ExprId, ExprId) {
    if is_zero_expr(w2, pool) {
        return (pool.integer(1), z);
    }
    if sign_hint(neg(w2, pool), positive, pool) == Sgn::Pos {
        let k = sqrt_pos(neg(w2, pool), pool);
        let kz = pool.mul(vec![k, z]);
        return (
            pool.func("cosh", vec![kz]),
            div(pool.func("sinh", vec![kz]), k, pool),
        );
    }
    let w = sqrt_pos(w2, pool);
    let wz = pool.mul(vec![w, z]);
    (
        pool.func("cos", vec![wz]),
        div(pool.func("sin", vec![wz]), w, pool),
    )
}

/// Duhamel's integral `∫_{lo}^{hi} body dτ`: `integrate` with the
/// antiderivative checked exactly, else an unevaluated `Integral`.
pub(crate) fn definite_or_placeholder(
    body: ExprId,
    tau: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> (ExprId, bool) {
    if let Some(v) = definite_closed(body, tau, lo, hi, pool) {
        return (v, true);
    }
    (
        super::project::integral_placeholder(body, tau, lo, hi, pool),
        false,
    )
}

/// `∫_{lo}^{hi} body dτ` by an exactly checked antiderivative.
pub(crate) fn definite_closed(
    body: ExprId,
    tau: ExprId,
    lo: ExprId,
    hi: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let body = simp(body, pool);
    if body == pool.integer(0) {
        return Some(body);
    }
    let anti = super::project::antiderivative(body, tau, pool)?;
    let v = sub(subs1(anti, tau, hi, pool), subs1(anti, tau, lo, pool), pool);
    let v = simp(v, pool);
    if super::project::has_singular_value(v, pool) {
        return None;
    }
    Some(v)
}

fn fresh_tau(avoid: &[ExprId], pool: &ExprPool) -> ExprId {
    super::super::sturm_liouville::fresh_symbol("tau", Domain::Real, avoid, pool)
}

/// Solve the mode problem `a₂T'' + a₁T' + a₀T = q` with `data`.
pub(crate) fn solve_mode(
    m: ModeOde,
    s: ExprId,
    data: &SData,
    q: ExprId,
    positive: &[ExprId],
    const_names: &mut Vec<String>,
    pool: &ExprPool,
) -> Result<ModeSolution, PdeError> {
    let mut sol = ModeSolution {
        value: pool.integer(0),
        closed: true,
        side_conditions: Vec::new(),
        notes: Vec::new(),
        constants: Vec::new(),
        form: "constant_coefficients",
    };
    let q = simp(q, pool);
    let has_q = q != pool.integer(0);
    match data {
        SData::Initial { t0, values } => {
            let coeffs_const = [m.a2, m.a1, m.a0]
                .iter()
                .all(|&c| !mentions_var(c, s, pool));
            if !coeffs_const {
                return Err(PdeError::SeparationModeOde(format!(
                    "the amplitude equation in {} has variable coefficients",
                    pool.display(s)
                )));
            }
            let z = sub(s, *t0, pool);
            let tau = fresh_tau(&[q, m.a0, m.a1, m.a2, s, *t0], pool);
            match m.order {
                1 => {
                    let rho = simp(div(m.a0, m.a1, pool), pool);
                    let v0 = values.first().copied().unwrap_or(pool.integer(0));
                    let hom = pool.mul(vec![
                        v0,
                        pool.func("exp", vec![neg(pool.mul(vec![rho, z]), pool)]),
                    ]);
                    let mut val = hom;
                    if has_q {
                        let kern = pool.func(
                            "exp",
                            vec![neg(pool.mul(vec![rho, sub(s, tau, pool)]), pool)],
                        );
                        let body = simp(
                            pool.mul(vec![kern, div(subs1(q, s, tau, pool), m.a1, pool)]),
                            pool,
                        );
                        let (v, closed) = definite_or_placeholder(body, tau, *t0, s, pool);
                        sol.closed &= closed;
                        val = pool.add(vec![val, v]);
                    }
                    sol.value = simp(val, pool);
                }
                2 => {
                    let sigma = simp(div(m.a1, pool.mul(vec![pool.integer(2), m.a2]), pool), pool);
                    let w2 = simp(
                        sub(
                            div(m.a0, m.a2, pool),
                            pool.pow(sigma, pool.integer(2)),
                            pool,
                        ),
                        pool,
                    );
                    let (c, sfun) = cs(z, w2, positive, pool);
                    let v0 = values.first().copied().unwrap_or(pool.integer(0));
                    let v1 = values.get(1).copied().unwrap_or(pool.integer(0));
                    let damp = pool.func("exp", vec![neg(pool.mul(vec![sigma, z]), pool)]);
                    let mut val = pool.mul(vec![
                        damp,
                        pool.add(vec![
                            pool.mul(vec![v0, c]),
                            pool.mul(vec![pool.add(vec![v1, pool.mul(vec![sigma, v0])]), sfun]),
                        ]),
                    ]);
                    if sigma == pool.integer(0) {
                        val = pool.add(vec![pool.mul(vec![v0, c]), pool.mul(vec![v1, sfun])]);
                    }
                    if has_q {
                        let zt = sub(s, tau, pool);
                        let (_, st) = cs(zt, w2, positive, pool);
                        let kern = pool.mul(vec![
                            pool.func("exp", vec![neg(pool.mul(vec![sigma, zt]), pool)]),
                            st,
                        ]);
                        let body = simp(
                            pool.mul(vec![kern, div(subs1(q, s, tau, pool), m.a2, pool)]),
                            pool,
                        );
                        let (v, closed) = definite_or_placeholder(body, tau, *t0, s, pool);
                        sol.closed &= closed;
                        val = pool.add(vec![val, v]);
                    }
                    if !is_zero_expr(w2, pool) && sign_hint(w2, positive, pool) == Sgn::Unknown {
                        let ne = pool.predicate(
                            crate::kernel::expr::PredicateKind::Ne,
                            vec![w2, pool.integer(0)],
                        );
                        sol.side_conditions.push(ne);
                    }
                    sol.value = simp(val, pool);
                }
                _ => {
                    return Err(PdeError::SeparationModeOde(
                        "the amplitude equation has no derivative in the evolution variable".into(),
                    ))
                }
            }
            Ok(sol)
        }
        SData::TwoPoint {
            s0,
            s1,
            left,
            right,
        } => {
            if m.order != 2 {
                return Err(PdeError::SeparationModeOde(format!(
                    "a boundary-value problem in {} needs a second-order amplitude equation",
                    pool.display(s)
                )));
            }
            let p = simp(div(m.a1, m.a2, pool), pool);
            let qq = simp(div(m.a0, m.a2, pool), pool);
            let q_src = simp(div(q, m.a2, pool), pool);
            if has_q && matches!(left, SEnd::Bounded) {
                return Err(PdeError::SeparationModeOde(
                    "a source term with a singular end in the data direction is not supported \
                     (pose the source with homogeneous boundary data: it is then expanded in \
                     eigenfunctions in every direction)"
                        .into(),
                ));
            }
            let (phi1, phi2, form, bounded_ok) =
                basis_two_point(p, qq, s, *s0, *s1, positive, pool)?;
            sol.form = form;
            // Particular solution by variation of parameters.
            let mut tp = pool.integer(0);
            if has_q {
                let tau = fresh_tau(&[q, s, *s0, *s1, phi1, phi2], pool);
                let w = simp(
                    sub(
                        pool.mul(vec![phi1, d(phi2, s, pool)?]),
                        pool.mul(vec![phi2, d(phi1, s, pool)?]),
                        pool,
                    ),
                    pool,
                );
                if is_zero_expr(w, pool) {
                    return Err(PdeError::SeparationModeOde(
                        "degenerate basis for the amplitude equation".into(),
                    ));
                }
                // G(s, τ) = (φ1(τ)φ2(s) − φ1(s)φ2(τ)) / W(τ)
                let at = |e: ExprId| subs1(e, s, tau, pool);
                let g = div(
                    sub(
                        pool.mul(vec![at(phi1), phi2]),
                        pool.mul(vec![phi1, at(phi2)]),
                        pool,
                    ),
                    at(w),
                    pool,
                );
                let body = simp(pool.mul(vec![g, at(q_src)]), pool);
                let lo = match left {
                    SEnd::Bounded => *s0,
                    SEnd::Face { at, .. } => *at,
                };
                let (v, closed) = definite_or_placeholder(body, tau, lo, s, pool);
                sol.closed &= closed;
                tp = v;
            }
            // Conditions: functionals ℓ(φ) = α φ(at) + β φ'(at).
            let functional =
                |end: &SEnd, f: ExprId| -> Result<Option<(ExprId, ExprId)>, PdeError> {
                    match end {
                        SEnd::Bounded => Ok(None),
                        SEnd::Face {
                            at,
                            alpha,
                            beta,
                            data,
                        } => {
                            let v = simp(
                                pool.add(vec![
                                    pool.mul(vec![*alpha, subs1(f, s, *at, pool)]),
                                    pool.mul(vec![*beta, subs1(d(f, s, pool)?, s, *at, pool)]),
                                ]),
                                pool,
                            );
                            Ok(Some((v, *data)))
                        }
                    }
                };
            let bounded_left = matches!(left, SEnd::Bounded);
            if matches!(right, SEnd::Bounded) {
                return Err(PdeError::SeparationModeOde(
                    "boundedness is supported at the left (singular) end only".into(),
                ));
            }
            if bounded_left {
                if !bounded_ok {
                    return Err(PdeError::SeparationModeOde(format!(
                        "no bounded solution selected at {} = {}",
                        pool.display(s),
                        pool.display(*s0)
                    )));
                }
                // phi1 is the bounded solution.
                let (m1, b1) = functional(right, phi1)?.expect("face");
                let (mp, _) = functional(right, tp)?.unwrap_or((pool.integer(0), b1));
                let rhs = simp(sub(b1, mp, pool), pool);
                if is_zero_expr(m1, pool) {
                    // Compatibility: rhs must vanish; then A is arbitrary.
                    if is_zero_expr(rhs, pool) {
                        let c = fresh_const(const_names, pool);
                        sol.constants.push(c);
                        sol.value = simp(pool.add(vec![pool.mul(vec![c, phi1]), tp]), pool);
                        sol.notes.push(
                            "this mode is determined only up to an additive multiple (the \
                             Neumann problem's constant)"
                                .into(),
                        );
                        return Ok(sol);
                    }
                    return Err(PdeError::SeparationCompatibility(format!(
                        "the data violates the compatibility condition {} = 0",
                        pool.display(rhs)
                    )));
                }
                sol.value = simp(
                    pool.add(vec![div(pool.mul(vec![rhs, phi1]), m1, pool), tp]),
                    pool,
                );
                return Ok(sol);
            }
            let (m00, b0) = functional(left, phi1)?.expect("face");
            let (m01, _) = functional(left, phi2)?.expect("face");
            let (m10, b1) = functional(right, phi1)?.expect("face");
            let (m11, _) = functional(right, phi2)?.expect("face");
            let (tp0, _) = functional(left, tp)?.expect("face");
            let (tp1, _) = functional(right, tp)?.expect("face");
            let r0 = simp(sub(b0, tp0, pool), pool);
            let r1 = simp(sub(b1, tp1, pool), pool);
            let det = simp(
                sub(pool.mul(vec![m00, m11]), pool.mul(vec![m01, m10]), pool),
                pool,
            );
            if !is_zero_expr(det, pool) {
                let a = div(
                    sub(pool.mul(vec![r0, m11]), pool.mul(vec![r1, m01]), pool),
                    det,
                    pool,
                );
                let b = div(
                    sub(pool.mul(vec![m00, r1]), pool.mul(vec![m10, r0]), pool),
                    det,
                    pool,
                );
                let a = simp(a, pool);
                let b = simp(b, pool);
                sol.value = simp(
                    pool.add(vec![pool.mul(vec![a, phi1]), pool.mul(vec![b, phi2]), tp]),
                    pool,
                );
                if sign_hint(det, positive, pool) == Sgn::Unknown {
                    sol.side_conditions.push(pool.predicate(
                        crate::kernel::expr::PredicateKind::Ne,
                        vec![det, pool.integer(0)],
                    ));
                }
                return Ok(sol);
            }
            // Rank ≤ 1: a particular combination plus the null vector.
            let rows = [(m00, m01, r0), (m10, m11, r1)];
            let pick = rows
                .iter()
                .enumerate()
                .flat_map(|(i, r)| [(i, 0usize, r.0), (i, 1usize, r.1)])
                .find(|&(_, _, c)| !is_zero_expr(c, pool));
            let Some((i, j, cij)) = pick else {
                return Err(PdeError::SeparationModeOde(
                    "the conditions do not constrain the amplitude".into(),
                ));
            };
            let (ri, other) = (rows[i].2, if j == 0 { rows[i].1 } else { rows[i].0 });
            let (pj, po) = if j == 0 { (phi1, phi2) } else { (phi2, phi1) };
            let part = div(pool.mul(vec![ri, pj]), cij, pool);
            let null = simp(
                sub(pool.mul(vec![cij, po]), pool.mul(vec![other, pj]), pool),
                pool,
            );
            let k = 1 - i;
            let (mk0, mk1, rk) = rows[k];
            let (a_part, b_part) = if j == 0 {
                (div(ri, cij, pool), pool.integer(0))
            } else {
                (pool.integer(0), div(ri, cij, pool))
            };
            let compat = simp(
                sub(
                    pool.add(vec![
                        pool.mul(vec![mk0, a_part]),
                        pool.mul(vec![mk1, b_part]),
                    ]),
                    rk,
                    pool,
                ),
                pool,
            );
            if !is_zero_expr(compat, pool) {
                let st = zero_test(compat, pool, None).status;
                if st == ZeroStatus::ProvedNonzero {
                    return Err(PdeError::SeparationCompatibility(format!(
                        "the boundary data violates the compatibility condition {} = 0",
                        pool.display(compat)
                    )));
                }
                sol.side_conditions.push(pool.predicate(
                    crate::kernel::expr::PredicateKind::Eq,
                    vec![compat, pool.integer(0)],
                ));
            }
            let c = fresh_const(const_names, pool);
            sol.constants.push(c);
            sol.notes.push(
                "the zero mode is determined only up to a multiple of the null solution (an \
                 arbitrary constant)"
                    .into(),
            );
            sol.value = simp(pool.add(vec![part, pool.mul(vec![c, null]), tp]), pool);
            Ok(sol)
        }
    }
}

fn fresh_const(names: &mut Vec<String>, pool: &ExprPool) -> ExprId {
    let n = super::separate::fresh_name("C1", names);
    pool.symbol(n, Domain::Real)
}

/// A basis `(φ₁, φ₂)` of `T'' + pT' + qT = 0` on `[s0, s1]` and the form;
/// when the left end is singular (`s0 = 0`) `φ₁` is the bounded solution
/// and the flag is set.
fn basis_two_point(
    p: ExprId,
    q: ExprId,
    s: ExprId,
    s0: ExprId,
    s1: ExprId,
    positive: &[ExprId],
    pool: &ExprPool,
) -> Result<(ExprId, ExprId, &'static str, bool), PdeError> {
    let free = |e: ExprId| !mentions_var(e, s, pool);
    if free(p) && free(q) {
        let sigma = simp(div(p, pool.integer(2), pool), pool);
        let w2 = simp(sub(q, pool.pow(sigma, pool.integer(2)), pool), pool);
        let damp = pool.func(
            "exp",
            vec![neg(pool.mul(vec![sigma, sub(s, s0, pool)]), pool)],
        );
        let (_, s_a) = cs(sub(s1, s, pool), w2, positive, pool);
        let (_, s_b) = cs(sub(s, s0, pool), w2, positive, pool);
        let (f1, f2) = if sigma == pool.integer(0) {
            (s_a, s_b)
        } else {
            (pool.mul(vec![damp, s_a]), pool.mul(vec![damp, s_b]))
        };
        return Ok((
            simp(f1, pool),
            simp(f2, pool),
            "constant_coefficients",
            false,
        ));
    }
    let ps = simp(super::separate::expanded(pool.mul(vec![p, s]), pool), pool);
    let qs2 = simp(
        super::separate::expanded(pool.mul(vec![q, pool.pow(s, pool.integer(2))]), pool),
        pool,
    );
    let at_zero = s0 == pool.integer(0);
    if free(ps) && free(qs2) {
        // Euler: p² + (β − 1)p + γ = 0.
        let beta = ps;
        let gamma = qs2;
        let disc = simp(
            sub(
                pool.pow(sub(beta, pool.integer(1), pool), pool.integer(2)),
                pool.mul(vec![pool.integer(4), gamma]),
                pool,
            ),
            pool,
        );
        let mid = simp(
            div(sub(pool.integer(1), beta, pool), pool.integer(2), pool),
            pool,
        );
        if is_zero_expr(disc, pool) {
            let f1 = pool.pow(s, mid);
            let f2 = pool.mul(vec![pool.pow(s, mid), pool.func("log", vec![s])]);
            let ok = at_zero && sign_hint(mid, positive, pool) != Sgn::Neg;
            return Ok((simp(f1, pool), simp(f2, pool), "euler", ok));
        }
        let root = sqrt_pos(disc, pool);
        let half_root = div(root, pool.integer(2), pool);
        let p_plus = simp(pool.add(vec![mid, half_root]), pool);
        let p_minus = simp(sub(mid, half_root, pool), pool);
        let (a, b) = if sign_hint(p_plus, positive, pool) == Sgn::Neg {
            (p_minus, p_plus)
        } else {
            (p_plus, p_minus)
        };
        let ok = at_zero && sign_hint(a, positive, pool) != Sgn::Neg;
        return Ok((
            simp(pool.pow(s, a), pool),
            simp(pool.pow(s, b), pool),
            "euler",
            ok,
        ));
    }
    // Bessel / spherical Bessel: p = 1/s or 2/s, q = A + B/s².
    let two_a = simp(d(d(qs2, s, pool)?, s, pool)?, pool);
    let big_a = simp(div(two_a, pool.integer(2), pool), pool);
    let big_b = simp(subs1(qs2, s, pool.integer(0), pool), pool);
    let poly_ok = free(big_a)
        && free(big_b)
        && is_zero_expr(
            sub(
                qs2,
                pool.add(vec![
                    pool.mul(vec![big_a, pool.pow(s, pool.integer(2))]),
                    big_b,
                ]),
                pool,
            ),
            pool,
        );
    if poly_ok && (ps == pool.integer(1) || ps == pool.integer(2)) {
        let spherical = ps == pool.integer(2);
        let nu2 = if spherical {
            simp(sub(pool.rational(1, 4), big_b, pool), pool)
        } else {
            simp(neg(big_b, pool), pool)
        };
        let nu = sqrt_pos(nu2, pool);
        let pre = if spherical {
            pool.pow(s, pool.rational(-1, 2))
        } else {
            pool.integer(1)
        };
        let (f1, f2) = if sign_hint(neg(big_a, pool), positive, pool) == Sgn::Pos {
            let k = sqrt_pos(neg(big_a, pool), pool);
            let ks = pool.mul(vec![k, s]);
            (
                pool.func("bessel_i", vec![nu, ks]),
                pool.func("bessel_k", vec![nu, ks]),
            )
        } else {
            let w = sqrt_pos(big_a, pool);
            let ws = pool.mul(vec![w, s]);
            (
                pool.func("bessel_j", vec![nu, ws]),
                pool.func("bessel_y", vec![nu, ws]),
            )
        };
        let form = if spherical {
            "spherical_bessel"
        } else {
            "bessel"
        };
        return Ok((
            simp(pool.mul(vec![pre, f1]), pool),
            simp(pool.mul(vec![pre, f2]), pool),
            form,
            at_zero,
        ));
    }
    Err(PdeError::SeparationModeOde(format!(
        "the amplitude equation T'' + ({})T' + ({})T = 0 is not of constant-coefficient, Euler \
         or Bessel type",
        pool.display(p),
        pool.display(q)
    )))
}
