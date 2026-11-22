//! Second-order linear equations solved by Airy and Bessel functions, and the
//! second-order equations without `y` solved by reduction of order.
//!
//! # Airy
//!
//! `y'' + p·y' + q(x)·y = 0` with a constant `p` and `q − p²/4 = −(a·x + b)`,
//! `a ≠ 0`: `y = e^{−px/2}·{Ai, Bi}(k·(x + b/a))` with `k = a^{1/3}` (real cube
//! root), because `Ai''(t) = t·Ai(t)` and `k³ = a`.
//!
//! # Bessel
//!
//! The family reducible to Bessel's equation by `y = x^α·u(β·x^γ)` (Kamke
//! 2.162; SymPy's `2nd_linear_bessel`):
//!
//! `x²·y'' + (1 − 2α)·x·y' + (±β²γ²·x^{2γ} + α² − ν²γ²)·y = 0`
//!
//! has the fundamental system `x^α·J_ν(βx^γ), x^α·Y_ν(βx^γ)` (sign `+`) and
//! `x^α·I_ν(βx^γ), x^α·K_ν(βx^γ)` (sign `−`). Recognition reads `x·P` and
//! `x²·Q` of the monic equation `y'' + P·y' + Q·y = 0` as a constant `1 − 2α`
//! and `A·x^s + D` with rational literals; then `γ = s/2`, `β = √|A|/|γ|`,
//! `ν = √((α² − D)/γ²)` (a negative `ν²` — an imaginary order — is declined).
//!
//! Both bases go through [`super::constant_coeff`]'s `compose_from_basis`, so
//! a forcing term is handled by variation of parameters and **every** returned
//! solution is verified by substitution like any other class.
//!
//! # `y` absent: `F(x, y', y'') = 0`
//!
//! With `p = y'` this is a first-order equation `F(x, p, p') = 0`, solved by
//! the first-order cascade; an explicit `p(x)` integrates to
//! `y = ∫p dx + C`. (`y'' = y'²` gives `p = 1/(C₁ − x)` and
//! `y = C₂ − log(C₁ − x)`.) The final `y` is verified against the original
//! second-order equation.

use rug::{Integer, Rational};

use super::{
    contains, ddx, div, integrate_or_decline, residual_is_zero, simp, subs1, ConstGen,
    DsolveBranch, DsolveError, OdeInput,
};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};

/// A rational literal.
fn lit(e: ExprId, pool: &ExprPool) -> Option<Rational> {
    match pool.get(e) {
        ExprData::Integer(n) => Some(Rational::from(n.0.clone())),
        ExprData::Rational(r) => Some(r.0.clone()),
        _ => None,
    }
}

fn rat(q: &Rational, pool: &ExprPool) -> ExprId {
    if *q.denom() == 1 {
        pool.integer(q.numer().clone())
    } else {
        pool.rational(q.numer().clone(), q.denom().clone())
    }
}

/// `e` as `Σ c·x^r` with rational literals `c`, `r` (like powers merged).
fn monomials(e: ExprId, x: ExprId, pool: &ExprPool) -> Option<Vec<(Rational, Rational)>> {
    let terms: Vec<ExprId> = match pool.get(e) {
        ExprData::Add(xs) => xs.to_vec(),
        _ => vec![e],
    };
    let mut out: Vec<(Rational, Rational)> = Vec::new();
    for t in terms {
        let factors: Vec<ExprId> = match pool.get(t) {
            ExprData::Mul(xs) => xs.to_vec(),
            _ => vec![t],
        };
        let mut c = Rational::from(1);
        let mut r = Rational::from(0);
        for f in factors {
            if let Some(v) = lit(f, pool) {
                c *= v;
            } else if f == x {
                r += 1;
            } else if let ExprData::Pow { base, exp } = pool.get(f) {
                if base != x {
                    return None;
                }
                r += lit(exp, pool)?;
            } else {
                return None;
            }
        }
        if let Some(slot) = out.iter_mut().find(|(_, rr)| *rr == r) {
            slot.0 += c;
        } else {
            out.push((c, r));
        }
    }
    out.retain(|(c, _)| *c != 0);
    Some(out)
}

/// `√q` as an exact rational when `q ≥ 0` is a square.
fn rational_sqrt(q: &Rational) -> Option<Rational> {
    if *q < 0 {
        return None;
    }
    let (n, d) = (q.numer(), q.denom());
    if n.is_perfect_square() && d.is_perfect_square() {
        Some(Rational::from((n.clone().sqrt(), d.clone().sqrt())))
    } else {
        None
    }
}

/// `√q` for a rational `q ≥ 0`, exact when it can be.
fn sqrt_expr(q: &Rational, pool: &ExprPool) -> ExprId {
    match rational_sqrt(q) {
        Some(r) => rat(&r, pool),
        None => pool.pow(rat(q, pool), pool.rational(1, 2)),
    }
}

/// Real cube root of a rational, exact when it can be.
fn cbrt_expr(q: &Rational, pool: &ExprPool) -> ExprId {
    let neg = *q < 0;
    let a = Rational::from(q.abs_ref());
    let (n, d) = (a.numer().clone(), a.denom().clone());
    let (rn, rem_n) = n.clone().root_rem(Integer::new(), 3);
    let (rd, rem_d) = d.clone().root_rem(Integer::new(), 3);
    let mag = if rem_n == 0 && rem_d == 0 {
        rat(&Rational::from((rn, rd)), pool)
    } else {
        pool.pow(rat(&a, pool), pool.rational(1, 3))
    };
    if neg {
        pool.mul(vec![pool.integer(-1_i32), mag])
    } else {
        mag
    }
}

/// The monic coefficients `P`, `Q` of `y'' + P y' + Q y`.
fn monic(coeffs: &[ExprId], pool: &ExprPool) -> Option<(ExprId, ExprId)> {
    if coeffs.len() != 3 || super::is_zero(coeffs[2], pool) {
        return None;
    }
    let p = super::normalized(div(coeffs[1], coeffs[2], pool), pool);
    let q = super::normalized(div(coeffs[0], coeffs[2], pool), pool);
    Some((p, q))
}

/// Airy basis for `y'' + p y' + q y = 0` (see the module docs), or `None`.
pub(super) fn airy_basis(coeffs: &[ExprId], x: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let (p, q) = monic(coeffs, pool)?;
    let p = simp(p, pool);
    if contains(p, x, pool) {
        return None;
    }
    let p_lit = lit(p, pool)?;
    // q − p²/4 = −(a x + b)
    let shifted = simp(
        pool.add(vec![q, rat(&(-(p_lit.clone() * &p_lit) / 4), pool)]),
        pool,
    );
    let mons = monomials(shifted, x, pool)?;
    let mut a = Rational::new();
    let mut b = Rational::new();
    for (c, r) in mons {
        if r == 1 {
            a = -c;
        } else if r == 0 {
            b = -c;
        } else {
            return None;
        }
    }
    if a == 0 {
        return None;
    }
    let k = cbrt_expr(&a, pool);
    let shift = rat(&(b / a.clone()), pool);
    let arg = simp(pool.mul(vec![k, pool.add(vec![x, shift])]), pool);
    let damp = if p_lit == 0 {
        None
    } else {
        Some(pool.func(
            "exp",
            vec![simp(pool.mul(vec![rat(&(-p_lit / 2), pool), x]), pool)],
        ))
    };
    let wrap = |f: ExprId| match damp {
        Some(d) => pool.mul(vec![d, f]),
        None => f,
    };
    Some(vec![
        wrap(pool.func("airy_ai", vec![arg])),
        wrap(pool.func("airy_bi", vec![arg])),
    ])
}

/// Bessel basis for the family in the module docs, or `None`.
pub(super) fn bessel_basis(coeffs: &[ExprId], x: ExprId, pool: &ExprPool) -> Option<Vec<ExprId>> {
    let (p, q) = monic(coeffs, pool)?;
    let xp = simp(pool.mul(vec![x, p]), pool);
    let p1 = lit(xp, pool)?;
    let x2q = simp(pool.mul(vec![pool.pow(x, pool.integer(2_i32)), q]), pool);
    let mons = monomials(x2q, x, pool)?;
    let mut big_a: Option<(Rational, Rational)> = None;
    let mut d = Rational::new();
    for (c, r) in mons {
        if r == 0 {
            d = c;
        } else if big_a.is_none() {
            big_a = Some((c, r));
        } else {
            return None;
        }
    }
    let (a_coef, s) = big_a?;
    let alpha: Rational = (Rational::from(1) - p1) / 2;
    let gamma: Rational = s / 2;
    let gamma2: Rational = gamma.clone() * &gamma;
    let nu2: Rational = (alpha.clone() * &alpha - d) / gamma2.clone();
    if nu2 < 0 {
        return None;
    }
    let modified = a_coef < 0;
    let beta2: Rational = Rational::from(a_coef.abs_ref()) / gamma2;
    let nu = sqrt_expr(&nu2, pool);
    let beta = sqrt_expr(&beta2, pool);
    let xg = if gamma == 1 {
        x
    } else {
        pool.pow(x, rat(&gamma, pool))
    };
    let arg = simp(pool.mul(vec![beta, xg]), pool);
    let (f1, f2) = if modified {
        ("bessel_i", "bessel_k")
    } else {
        ("bessel_j", "bessel_y")
    };
    let pre = (alpha != 0).then(|| pool.pow(x, rat(&alpha, pool)));
    let wrap = |f: ExprId| match pre {
        Some(m) => pool.mul(vec![m, f]),
        None => f,
    };
    Some(vec![
        wrap(pool.func(f1, vec![nu, arg])),
        wrap(pool.func(f2, vec![nu, arg])),
    ])
}

/// `F(x, y', y'') = 0` with no `y`: `p = y'` reduces it to first order.
pub(super) fn solve_y_absent(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    if input.order() != 2 || contains(input.equation, input.y, pool) {
        return Ok(None);
    }
    let (yp, ypp) = (input.derivs[0], input.derivs[1]);
    if !contains(input.equation, ypp, pool) {
        return Ok(None);
    }
    let p = pool.symbol("__dsolve_p", Domain::Real);
    let (first, pp) = OdeInput::first_order(input.x, p, pool);
    let eq1 = subs1(subs1(input.equation, ypp, pp, pool), yp, p, pool);
    let first = first.with_equation(eq1);
    let mark = gen.checkpoint();
    let branches = match super::first_order::solve(&first, gen, pool) {
        Ok(b) => b,
        Err(_) => {
            gen.rollback(mark, pool);
            return Ok(None);
        }
    };
    for b in branches {
        let Some(p_of_x) = b.y_of_x() else {
            continue;
        };
        let Ok(y0) = integrate_or_decline(p_of_x, input.x, pool) else {
            continue;
        };
        let c2 = gen.fresh(pool);
        let y = simp(pool.add(vec![y0, c2]), pool);
        let mut constants = b.constants.clone();
        constants.push(c2);
        // The first-order solution was verified for p; verify y against the
        // second-order equation itself.
        if ddx(y, input.x, pool).is_err() {
            continue;
        }
        if residual_is_zero(input, y, &constants, pool).is_ok() {
            return Ok(Some(vec![DsolveBranch::explicit(
                y,
                constants,
                "reduction_of_order_y_absent",
            )]));
        }
    }
    gen.rollback(mark, pool);
    Ok(None)
}
