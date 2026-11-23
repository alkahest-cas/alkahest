//! Canonical forms by characteristic coordinates.
//!
//! The characteristics of `A u_xx + B u_xy + C u_yy + … = G` are the level
//! sets `φ(x, y) = const` of the first integrals of
//! `A dy² − B dx dy + C dx² = 0`, i.e. of the ODEs `dy/dx = λ` with
//! `A λ² − B λ + C = 0`.  With `ξ = φ`, `η = ψ` and `u(x, y) = w(ξ, η)` the
//! chain rule gives the new principal coefficients
//!
//! ```text
//! A' = A φx² + B φx φy + C φy²
//! B' = 2A φx ψx + B (φx ψy + φy ψx) + 2C φy ψy
//! C' = A ψx² + B ψx ψy + C ψy²
//! D' = A φxx + B φxy + C φyy + D φx + E φy        (E' likewise with ψ)
//! ```
//!
//! so that `A' = C' = 0` (hyperbolic, `w_ξη + … = 0`), `A' = B' = 0`
//! (parabolic, `w_ηη + … = 0`) or, for `α = Re φ`, `β = Im φ` of a complex
//! characteristic, `A' = C'`, `B' = 0` (elliptic, `w_αα + w_ββ + … = 0`).
//! Each of these identities is checked exactly before a form is returned.

use crate::kernel::{ExprId, ExprPool};
use crate::ode::dsolve::{dsolve_with, OdeInput, SolutionForm};
use crate::simplify::assumptions::AssumptionContext;

use super::super::classify::{PdeType, TypeRegion};
use super::super::error::PdeError;
use super::linear::Linear2;
use super::util::{
    conj, d, div, exp, free_of, free_of_all, integ, is_zero, linear_in, mentions_i, neg,
    nicer_integral, simp, split_xy, sqrt, sub, subs1, subs2, tidy,
};

/// The result of [`super::canonical_form`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CanonicalForm {
    /// Hyperbolic, parabolic or elliptic (the type on [`Self::region`]).
    pub pde_type: PdeType,
    /// The region of the plane the form is valid on, when the type varies.
    pub region: Option<TypeRegion>,
    /// The original independent variables `(x, y)`.
    pub old_variables: (ExprId, ExprId),
    /// The new independent variables (`ξ, η`, or `α, β` when elliptic).
    pub new_variables: (ExprId, ExprId),
    /// The transformation: `ξ = φ(x, y)`, `η = ψ(x, y)`.
    pub transformation: (ExprId, ExprId),
    /// `(x, y)` in terms of the new variables, when found.
    pub inverse: Option<(ExprId, ExprId)>,
    /// The new unknown `w(ξ, η)`; `u(x, y) = w(φ, ψ)`.
    pub unknown: ExprId,
    /// The canonical equation `w_ξη + a w_ξ + b w_η + c w − g` (read `= 0`),
    /// normalised so that the principal coefficient is one.
    pub equation: ExprId,
    /// `[A', B', C', D', E', F', G']` after normalisation (principal
    /// coefficient one), in the new variables when
    /// [`Self::in_new_variables`].
    pub coefficients: Vec<ExprId>,
    /// Whether every coefficient was expressed in the new variables (if not,
    /// some still mention `x, y`).
    pub in_new_variables: bool,
    /// The factor dividing the transformed principal coefficient (`B'`,
    /// `C'` or `A'`) before normalisation.
    pub multiplier: ExprId,
    /// Side conditions (the region, `φ_y ≠ 0`, …).
    pub conditions: Vec<ExprId>,
}

/// A first integral `φ` of `dy/dx = λ(x, y)`: `φ_x + λ φ_y = 0`, `φ_y ≠ 0`.
pub(crate) fn first_integral(
    lambda: ExprId,
    x: ExprId,
    y: ExprId,
    pool: &ExprPool,
) -> Option<ExprId> {
    let lambda = simp(lambda, pool);
    let candidates = first_integral_candidates(lambda, x, y, pool);
    let check = |phi: ExprId| -> bool {
        let (Ok(px), Ok(py)) = (d(phi, x, pool), d(phi, y, pool)) else {
            return false;
        };
        !is_zero(py, pool) && is_zero(pool.add(vec![px, pool.mul(vec![lambda, py])]), pool)
    };
    for phi in candidates {
        let phi = simp(phi, pool);
        let nice = nicer_integral(phi, pool);
        if check(nice) {
            return Some(nice);
        }
        if check(phi) {
            return Some(phi);
        }
    }
    None
}

fn first_integral_candidates(lambda: ExprId, x: ExprId, y: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    let mut out = Vec::new();
    // dy/dx = λ(x)
    if free_of(lambda, y, pool) {
        if let Some(i) = integ(lambda, x, pool) {
            out.push(sub(y, i, pool));
            return out;
        }
    }
    // dy/dx = λ(y)  ⇒  x − ∫ dy/λ
    if free_of(lambda, x, pool) && !is_zero(lambda, pool) {
        if let Some(i) = integ(pool.pow(lambda, pool.integer(-1)), y, pool) {
            out.push(sub(i, x, pool));
        }
    }
    // separable λ = p(x) q(y)
    let (p, q, r) = split_xy(lambda, x, y, pool);
    if r == pool.integer(1) && !free_of(q, y, pool) {
        if let (Some(iq), Some(ip)) = (
            integ(pool.pow(q, pool.integer(-1)), y, pool),
            integ(p, x, pool),
        ) {
            out.push(sub(iq, ip, pool));
        }
    }
    // linear λ = P(x) y + Q(x)
    if let Some((qq, pp)) = linear_in(lambda, y, pool) {
        if free_of(pp, y, pool) && free_of(qq, y, pool) {
            if let Some(ip) = integ(pp, x, pool) {
                let m = exp(neg(ip, pool), pool);
                if let Some(iq) = integ(simp(pool.mul(vec![qq, m]), pool), x, pool) {
                    out.push(sub(pool.mul(vec![y, m]), iq, pool));
                }
            }
        }
    }
    // homogeneous λ(y/x): v = y/x, ln x = ∫ dv/(λ(v) − v)
    {
        let v = pool.symbol("__alkahest_pde2_v", crate::kernel::Domain::Real);
        let lv = simp(subs1(lambda, y, pool.mul(vec![v, x]), pool), pool);
        if free_of(lv, x, pool) {
            let den = simp(sub(lv, v, pool), pool);
            if !is_zero(den, pool) {
                if let Some(i) = integ(pool.pow(den, pool.integer(-1)), v, pool) {
                    let yx = pool.mul(vec![y, pool.pow(x, pool.integer(-1))]);
                    let phi = sub(subs1(i, v, yx, pool), pool.func("log", vec![x]), pool);
                    out.push(phi);
                    // exponentiated form often simpler
                    out.push(pool.mul(vec![
                        exp(subs1(i, v, yx, pool), pool),
                        pool.pow(x, pool.integer(-1)),
                    ]));
                }
            }
        }
    }
    // dsolve fallback: invert the constant of the general solution
    let (mut input, yp) = OdeInput::first_order(x, y, pool);
    input.equation = sub(yp, lambda, pool);
    if let Ok(rep) = dsolve_with(&input, &AssumptionContext::new(), pool) {
        for br in rep.branches {
            if br.constants.len() != 1 {
                continue;
            }
            let c = br.constants[0];
            match br.form {
                SolutionForm::Explicit(f) => {
                    if let Some((p0, q0)) = linear_in(f, c, pool) {
                        if !is_zero(q0, pool) {
                            out.push(div(sub(y, p0, pool), q0, pool));
                        }
                    }
                }
                SolutionForm::Implicit(g) => {
                    if let Some((p0, q0)) = linear_in(g, c, pool) {
                        if !is_zero(q0, pool) {
                            out.push(div(neg(p0, pool), q0, pool));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Transformed coefficients `[A', B', C', D', E']` under `ξ = φ`, `η = ψ`.
pub(crate) fn transformed(
    l: &Linear2,
    phi: ExprId,
    psi: ExprId,
    pool: &ExprPool,
) -> Result<[ExprId; 5], PdeError> {
    let (x, y) = (l.x, l.y);
    let [a, b, c, dd, e, _] = l.coef;
    let px = d(phi, x, pool)?;
    let py = d(phi, y, pool)?;
    let qx = d(psi, x, pool)?;
    let qy = d(psi, y, pool)?;
    let two = pool.integer(2);
    let a1 = simp(
        pool.add(vec![
            pool.mul(vec![a, px, px]),
            pool.mul(vec![b, px, py]),
            pool.mul(vec![c, py, py]),
        ]),
        pool,
    );
    let b1 = simp(
        pool.add(vec![
            pool.mul(vec![two, a, px, qx]),
            pool.mul(vec![
                b,
                pool.add(vec![pool.mul(vec![px, qy]), pool.mul(vec![py, qx])]),
            ]),
            pool.mul(vec![two, c, py, qy]),
        ]),
        pool,
    );
    let c1 = simp(
        pool.add(vec![
            pool.mul(vec![a, qx, qx]),
            pool.mul(vec![b, qx, qy]),
            pool.mul(vec![c, qy, qy]),
        ]),
        pool,
    );
    let lower = |f: ExprId, fx: ExprId, fy: ExprId| -> Result<ExprId, PdeError> {
        Ok(simp(
            pool.add(vec![
                pool.mul(vec![a, d(fx, x, pool)?]),
                pool.mul(vec![b, d(fx, y, pool)?]),
                pool.mul(vec![c, d(fy, y, pool)?]),
                pool.mul(vec![dd, fx]),
                pool.mul(vec![e, fy]),
            ]),
            pool,
        ))
        .map(|r| {
            let _ = f;
            r
        })
    };
    let d1 = lower(phi, px, py)?;
    let e1 = lower(psi, qx, qy)?;
    Ok([a1, b1, c1, d1, e1])
}

/// `(x, y)` as functions of `(ξ, η)` from `ξ = φ(x, y)`, `η = ψ(x, y)`:
/// solve one equation for a variable it is linear in, substitute, and solve
/// the other for the remaining variable (linear, or a pure power).  Checked
/// by substituting back.
pub(crate) fn invert(
    phi: ExprId,
    psi: ExprId,
    x: ExprId,
    y: ExprId,
    xi: ExprId,
    eta: ExprId,
    pool: &ExprPool,
) -> Option<(ExprId, ExprId)> {
    let tries = [(phi, xi, psi, eta), (psi, eta, phi, xi)];
    for (first, fv, second, sv) in tries {
        for (v1, v2) in [(y, x), (x, y)] {
            let v1_of = match linear_in(first, v1, pool) {
                Some((p, q))
                    if !is_zero(q, pool) && free_of(p, v1, pool) && free_of(q, v1, pool) =>
                {
                    div(sub(fv, p, pool), q, pool) // in v2 and fv
                }
                _ => match solve_power(first, v1, fv, pool) {
                    Some(r) => r,
                    None => continue,
                },
            };
            let g = tidy(subs1(second, v1, v1_of, pool), pool); // in v2, fv
            let v2_of = if free_of(g, v2, pool) {
                continue;
            } else if let Some((p2, q2)) = linear_in(g, v2, pool) {
                if is_zero(q2, pool) {
                    continue;
                }
                div(sub(sv, p2, pool), q2, pool)
            } else if let Some(r) = solve_power(g, v2, sv, pool) {
                r
            } else {
                continue;
            };
            let v1_final = simp(subs1(v1_of, v2, v2_of, pool), pool);
            let (xs, ys) = if v1 == y {
                (v2_of, v1_final)
            } else {
                (v1_final, v2_of)
            };
            if !free_of_all(xs, &[x, y], pool) || !free_of_all(ys, &[x, y], pool) {
                continue;
            }
            let ok1 = is_zero(sub(subs2(phi, (x, xs), (y, ys), pool), xi, pool), pool);
            let ok2 = is_zero(sub(subs2(psi, (x, xs), (y, ys), pool), eta, pool), pool);
            if ok1 && ok2 {
                return Some((xs, ys));
            }
        }
    }
    None
}

/// Solve `g(v) = target` where `g = κ · v^m` (κ free of `v`, `m` an integer
/// or rational): `v = (target/κ)^{1/m}`.
fn solve_power(g: ExprId, v: ExprId, target: ExprId, pool: &ExprPool) -> Option<ExprId> {
    // g = p + κ v^m: m = 1 + v g_vv / g_v must be constant
    let gv = d(g, v, pool).ok()?;
    if is_zero(gv, pool) {
        return None;
    }
    let gvv = d(gv, v, pool).ok()?;
    let m = tidy(
        pool.add(vec![
            pool.integer(1),
            pool.mul(vec![v, gvv, pool.pow(gv, pool.integer(-1))]),
        ]),
        pool,
    );
    if !free_of(m, v, pool) || is_zero(m, pool) {
        return None;
    }
    let kappa = tidy(
        pool.mul(vec![
            gv,
            pool.pow(m, pool.integer(-1)),
            pool.pow(v, sub(pool.integer(1), m, pool)),
        ]),
        pool,
    );
    if !free_of(kappa, v, pool) {
        return None;
    }
    let p = tidy(sub(g, pool.mul(vec![kappa, pool.pow(v, m)]), pool), pool);
    if !free_of(p, v, pool) {
        return None;
    }
    Some(tidy(
        pool.pow(
            pool.mul(vec![
                sub(target, p, pool),
                pool.pow(kappa, pool.integer(-1)),
            ]),
            pool.pow(m, pool.integer(-1)),
        ),
        pool,
    ))
}

/// Express `c(x, y)` as `g(ξ, η)` with `g(φ, ψ) = c`: directly when `c`
/// is free of `x, y`, else through the inverse, checked by substituting
/// back.
pub(crate) fn express(
    c: ExprId,
    l: &Linear2,
    phi: ExprId,
    psi: ExprId,
    xi: ExprId,
    eta: ExprId,
    inverse: Option<(ExprId, ExprId)>,
    pool: &ExprPool,
) -> Option<ExprId> {
    let (x, y) = (l.x, l.y);
    if free_of_all(c, &[x, y], pool) {
        return Some(c);
    }
    let (xs, ys) = inverse?;
    let g = simp(subs2(c, (x, xs), (y, ys), pool), pool);
    if !free_of_all(g, &[x, y], pool) {
        return None;
    }
    let back = subs2(g, (xi, phi), (eta, psi), pool);
    is_zero(sub(back, c, pool), pool).then_some(g)
}

/// Characteristic coordinates `(φ, ψ)` for a given type.
pub(crate) fn characteristic_coordinates(
    l: &Linear2,
    pde_type: PdeType,
    conds: &mut Vec<ExprId>,
    pool: &ExprPool,
) -> Result<(ExprId, ExprId), PdeError> {
    let (x, y) = (l.x, l.y);
    let [a, b, c, ..] = l.coef;
    let fail = |what: &str| {
        PdeError::CharacteristicsNotIntegrated(format!(
            "{what} for {} u_xx + {} u_xy + {} u_yy",
            pool.display(a),
            pool.display(b),
            pool.display(c)
        ))
    };
    let disc = simp(
        sub(
            pool.pow(b, pool.integer(2)),
            pool.mul(vec![pool.integer(4), a, c]),
            pool,
        ),
        pool,
    );
    let fi = |lam: ExprId| first_integral(lam, x, y, pool);
    match pde_type {
        PdeType::Hyperbolic => {
            if is_zero(a, pool) {
                // dx = 0 and B dy = C dx
                let phi2 =
                    fi(div(c, b, pool)).ok_or_else(|| fail("no first integral of dy/dx = C/B"))?;
                super::first_order::note_nonzero(b, conds, pool);
                Ok((x, phi2))
            } else if is_zero(c, pool) {
                let phi2 = first_integral(div(a, b, pool), y, x, pool)
                    .ok_or_else(|| fail("no first integral of dx/dy = A/B"))?;
                super::first_order::note_nonzero(b, conds, pool);
                Ok((y, phi2))
            } else {
                let sq = sqrt(disc, pool);
                let two_a = pool.mul(vec![pool.integer(2), a]);
                let l1 = div(pool.add(vec![b, sq]), two_a, pool);
                let l2 = div(sub(b, sq, pool), two_a, pool);
                let p1 = fi(l1).ok_or_else(|| {
                    fail(&format!(
                        "no first integral of dy/dx = {}",
                        pool.display(l1)
                    ))
                })?;
                let p2 = fi(l2).ok_or_else(|| {
                    fail(&format!(
                        "no first integral of dy/dx = {}",
                        pool.display(l2)
                    ))
                })?;
                super::first_order::note_nonzero(a, conds, pool);
                Ok((p1, p2))
            }
        }
        PdeType::Parabolic => {
            if is_zero(a, pool) {
                // then B = 0 too: C u_yy + … is already canonical
                return Ok((x, y));
            }
            let lam = div(b, pool.mul(vec![pool.integer(2), a]), pool);
            let phi = fi(lam).ok_or_else(|| {
                fail(&format!(
                    "no first integral of dy/dx = {}",
                    pool.display(lam)
                ))
            })?;
            super::first_order::note_nonzero(a, conds, pool);
            let py = d(phi, y, pool)?;
            let psi = if !is_zero(py, pool) { x } else { y };
            Ok((phi, psi))
        }
        PdeType::Elliptic => {
            if is_zero(a, pool) {
                return Err(fail("an elliptic operator with A = 0"));
            }
            let sq = sqrt(neg(disc, pool), pool);
            let i = pool.imaginary_unit();
            let lam = div(
                pool.add(vec![b, pool.mul(vec![i, sq])]),
                pool.mul(vec![pool.integer(2), a]),
                pool,
            );
            let phi = fi(lam).ok_or_else(|| {
                fail(&format!(
                    "no first integral of dy/dx = {}",
                    pool.display(lam)
                ))
            })?;
            let phi = tidy(phi, pool);
            let cphi = conj(phi, pool);
            let alpha = tidy(
                pool.mul(vec![pool.rational(1, 2), pool.add(vec![phi, cphi])]),
                pool,
            );
            let beta = tidy(
                pool.mul(vec![
                    pool.mul(vec![pool.rational(-1, 2), i]),
                    sub(phi, cphi, pool),
                ]),
                pool,
            );
            if mentions_i(alpha, pool) || mentions_i(beta, pool) {
                return Err(fail(
                    "the complex characteristic does not split into real and imaginary parts",
                ));
            }
            super::first_order::note_nonzero(a, conds, pool);
            Ok((alpha, beta))
        }
        _ => Err(PdeError::TypeNotUniform(format!(
            "no canonical form for type {}",
            pde_type.as_str()
        ))),
    }
}

/// Build the canonical form for `l` of the given type.
pub(crate) fn build(
    l: &Linear2,
    pde_type: PdeType,
    region: Option<TypeRegion>,
    pool: &ExprPool,
) -> Result<CanonicalForm, PdeError> {
    let mut conds = Vec::new();
    if let Some(r) = &region {
        conds.push(r.condition);
    }
    let (phi, psi) = characteristic_coordinates(l, pde_type, &mut conds, pool)?;
    let (phi, psi) = (tidy(phi, pool), tidy(psi, pool));
    let first = build_from(l, pde_type, region.clone(), phi, psi, conds.clone(), pool)?;
    let constant = |f: &CanonicalForm| {
        f.in_new_variables
            && f.coefficients[..6]
                .iter()
                .all(|&c| free_of_all(c, &[f.new_variables.0, f.new_variables.1], pool))
    };
    if constant(&first) || (first.in_new_variables && has_no_power_of_product(&first, pool)) {
        return Ok(first);
    }
    // A function of a first integral is one too: logarithmic coordinates
    // often make the coefficients constant (x² u_xx − y² u_yy with
    // ξ = log(y/x), η = log(xy)) where the inversion of ξ = y/x, η = xy
    // would need branches.
    let lphi = tidy(pool.func("log", vec![phi]), pool);
    let lpsi = tidy(pool.func("log", vec![psi]), pool);
    let mut fallback: Option<CanonicalForm> = None;
    for (p2, q2) in [(lphi, lpsi), (lphi, psi), (phi, lpsi)] {
        if let Ok(f) = build_from(l, pde_type, region.clone(), p2, q2, conds.clone(), pool) {
            if constant(&f) {
                return Ok(f);
            }
            if f.in_new_variables && fallback.is_none() {
                fallback = Some(f);
            }
        }
    }
    if first.in_new_variables {
        return Ok(first);
    }
    Ok(fallback.unwrap_or(first))
}

/// Whether the inverse map is free of fractional powers (no branch was
/// chosen to express the coefficients).
fn has_no_power_of_product(f: &CanonicalForm, pool: &ExprPool) -> bool {
    let Some((a, b)) = f.inverse else {
        return true;
    };
    let frac = |e: ExprId| {
        let mut seen = std::collections::HashSet::new();
        let mut stack = vec![e];
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            let r = pool.with(n, |d| match d {
                crate::kernel::ExprData::Pow { exp, .. } => {
                    Some(pool.with(*exp, |e| matches!(e, crate::kernel::ExprData::Rational(_))))
                }
                _ => None,
            });
            if r == Some(true) {
                return true;
            }
            super::super::jet::push_children(n, pool, &mut stack);
        }
        false
    };
    !frac(a) && !frac(b)
}

#[allow(clippy::too_many_arguments)]
fn build_from(
    l: &Linear2,
    pde_type: PdeType,
    region: Option<TypeRegion>,
    phi: ExprId,
    psi: ExprId,
    conds: Vec<ExprId>,
    pool: &ExprPool,
) -> Result<CanonicalForm, PdeError> {
    let [a1, b1, c1, d1, e1] = transformed(l, phi, psi, pool)?;
    let bad = |what: &str| {
        PdeError::CharacteristicsNotIntegrated(format!(
            "the characteristic coordinates ξ = {}, η = {} do not give the {} form ({what})",
            pool.display(phi),
            pool.display(psi),
            pde_type.as_str()
        ))
    };
    let jac = simp(
        sub(
            pool.mul(vec![d(phi, l.x, pool)?, d(psi, l.y, pool)?]),
            pool.mul(vec![d(phi, l.y, pool)?, d(psi, l.x, pool)?]),
            pool,
        ),
        pool,
    );
    if is_zero(jac, pool) {
        return Err(bad("the Jacobian vanishes"));
    }
    let lead = match pde_type {
        PdeType::Hyperbolic => {
            if !is_zero(a1, pool) || !is_zero(c1, pool) {
                return Err(bad("A' or C' does not vanish"));
            }
            b1
        }
        PdeType::Parabolic => {
            if !is_zero(a1, pool) || !is_zero(b1, pool) {
                return Err(bad("A' or B' does not vanish"));
            }
            c1
        }
        PdeType::Elliptic => {
            if !is_zero(sub(a1, c1, pool), pool) || !is_zero(b1, pool) {
                return Err(bad("A' ≠ C' or B' ≠ 0"));
            }
            a1
        }
        _ => return Err(bad("unsupported type")),
    };
    if is_zero(lead, pool) {
        return Err(bad("the principal coefficient vanishes"));
    }
    let mut taken = super::util::names_in(
        &[
            l.u, l.rhs, l.coef[0], l.coef[1], l.coef[2], l.coef[3], l.coef[4], l.coef[5],
        ],
        pool,
    );
    let (n1, n2) = if pde_type == PdeType::Elliptic {
        ("alpha", "beta")
    } else {
        ("xi", "eta")
    };
    let xi = super::util::fresh_symbol(n1, &mut taken, pool);
    let eta = super::util::fresh_symbol(n2, &mut taken, pool);
    let wname = super::util::fresh_name("w", &mut taken);
    let w = pool.func(wname, vec![xi, eta]);
    let inverse = invert(phi, psi, l.x, l.y, xi, eta, pool);
    let raw = [a1, b1, c1, d1, e1, l.f(), l.rhs];
    let mut coefficients = Vec::with_capacity(7);
    let mut in_new = true;
    for &r in &raw {
        let n = tidy(div(r, lead, pool), pool);
        match express(n, l, phi, psi, xi, eta, inverse, pool) {
            Some(g) => coefficients.push(tidy(g, pool)),
            None => {
                in_new = false;
                coefficients.push(n);
            }
        }
    }
    let dw = |e: ExprId, v: ExprId| {
        crate::diff::diff(e, v, pool)
            .map(|r| r.value)
            .map_err(|err| PdeError::NotRationalInJets(err.to_string()))
    };
    let wx = dw(w, xi)?;
    let wy = dw(w, eta)?;
    let parts = [dw(wx, xi)?, dw(wx, eta)?, dw(wy, eta)?, wx, wy, w];
    let mut terms: Vec<ExprId> = coefficients[..6]
        .iter()
        .zip(parts)
        .filter(|(&c, _)| !is_zero(c, pool))
        .map(|(&c, p)| pool.mul(vec![c, p]))
        .collect();
    if !is_zero(coefficients[6], pool) {
        terms.push(neg(coefficients[6], pool));
    }
    let equation = simp(pool.add(terms), pool);
    let _ = exp;
    Ok(CanonicalForm {
        pde_type,
        region,
        old_variables: (l.x, l.y),
        new_variables: (xi, eta),
        transformation: (phi, psi),
        inverse,
        unknown: w,
        equation,
        coefficients,
        in_new_variables: in_new,
        multiplier: lead,
        conditions: conds,
    })
}
