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
    contains, ddx, div, integrate_or_decline, residual_is_zero, simp, sub, subs1, ConstGen,
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

// ---------------------------------------------------------------------------
// `x` absent: F(y, y', y'') = 0 via p(y) = y'
// ---------------------------------------------------------------------------

/// `F(y, y', y'') = 0` with no `x`: with `p(y) = y'`, `y'' = p·dp/dy`, a
/// first-order equation in `p(y)`. An explicit `p = g(y)` gives
/// `x + C = ∫ dy/g(y)`, which is inverted for `y` when it can be (`y'' = −y`
/// gives `asin(y/√C₁) = x + C₂`); otherwise the relation is returned as an
/// implicit solution. Either form is verified against the original equation —
/// the implicit one by implicit differentiation, see
/// [`second_order_implicit_is_zero`].
pub(super) fn solve_x_absent(
    input: &OdeInput,
    gen: &mut ConstGen,
    pool: &ExprPool,
) -> Result<Option<Vec<DsolveBranch>>, DsolveError> {
    if input.order() != 2 || contains(input.equation, input.x, pool) {
        return Ok(None);
    }
    let (yp, ypp) = (input.derivs[0], input.derivs[1]);
    if !contains(input.equation, ypp, pool) || !contains(input.equation, input.y, pool) {
        return Ok(None);
    }
    let p = pool.symbol("__dsolve_p", Domain::Real);
    let (first, pp) = OdeInput::first_order(input.y, p, pool);
    let eq1 = subs1(
        subs1(input.equation, ypp, pool.mul(vec![p, pp]), pool),
        yp,
        p,
        pool,
    );
    let first = first.with_equation(simp(eq1, pool));
    let mark = gen.checkpoint();
    let branches = match super::first_order::solve(&first, gen, pool) {
        Ok(b) => b,
        Err(_e) => {
            #[cfg(test)]
            if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
                eprintln!("X_ABSENT first-order in p(y) failed: {_e}");
            }
            gen.rollback(mark, pool);
            return Ok(None);
        }
    };
    #[cfg(test)]
    if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
        for b in &branches {
            eprintln!("X_ABSENT p-branch {:?} {}", b.form, b.render(pool));
        }
    }
    let mut implicit: Option<DsolveBranch> = None;
    for b in branches {
        let Some(p_of_y) = b.y_of_x() else {
            continue;
        };
        if super::is_zero(p_of_y, pool) {
            continue;
        }
        let integrand = simp(
            super::fold_exp_log(div(pool.integer(1_i32), p_of_y, pool), pool),
            pool,
        );
        let big_i = match integrate_or_decline(integrand, input.y, pool) {
            Ok(v) => v,
            Err(_e) => {
                #[cfg(test)]
                if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
                    eprintln!("X_ABSENT ∫dy/p failed: {_e}");
                }
                continue;
            }
        };
        #[cfg(test)]
        if std::env::var_os("ALKAHEST_DSOLVE_TRACE").is_some() {
            eprintln!(
                "X_ABSENT I(y) = {} inverse = {:?}",
                pool.display(big_i),
                invert_for(big_i, input.y, input.x, pool).map(|e| pool.display(e).to_string())
            );
        }
        let c2 = gen.fresh(pool);
        let mut constants = b.constants.clone();
        constants.push(c2);
        let target = simp(pool.add(vec![input.x, c2]), pool);
        // Explicit: invert I(y) = x + C₂.
        if let Some(y) = invert_for(big_i, input.y, target, pool) {
            let y = simp(y, pool);
            if !contains(y, input.y, pool) && residual_is_zero(input, y, &constants, pool).is_ok() {
                return Ok(Some(vec![DsolveBranch::explicit(
                    y,
                    constants,
                    "autonomous_x_absent",
                )]));
            }
        }
        if implicit.is_none() {
            let relation = simp(sub(big_i, target, pool), pool);
            if second_order_implicit_is_zero(input, relation, &constants, pool).is_ok() {
                implicit = Some(DsolveBranch::implicit(
                    relation,
                    constants,
                    "autonomous_x_absent",
                ));
            }
        }
    }
    if let Some(b) = implicit {
        return Ok(Some(vec![b]));
    }
    gen.rollback(mark, pool);
    Ok(None)
}

/// Solve `h(y) = target` for `y` when `y` occurs once in `h`, peeling one
/// invertible operation at a time. Branch choices (a principal `asin`, a
/// positive root) are not decided here: the candidate is verified by the
/// caller.
fn invert_for(h: ExprId, y: ExprId, target: ExprId, pool: &ExprPool) -> Option<ExprId> {
    if h == y {
        return Some(target);
    }
    match pool.get(h) {
        ExprData::Add(xs) => {
            let (with, without): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&t| contains(t, y, pool));
            if with.len() != 1 {
                return None;
            }
            invert_for(with[0], y, sub(target, pool.add(without), pool), pool)
        }
        ExprData::Mul(xs) => {
            let (with, without): (Vec<ExprId>, Vec<ExprId>) =
                xs.iter().partition(|&&t| contains(t, y, pool));
            if with.len() != 1 {
                return None;
            }
            invert_for(with[0], y, div(target, pool.mul(without), pool), pool)
        }
        ExprData::Pow { base, exp } => {
            if !contains(exp, y, pool) {
                let inv = div(pool.integer(1_i32), exp, pool);
                invert_for(base, y, pool.pow(target, inv), pool)
            } else if !contains(base, y, pool) {
                let t = div(
                    pool.func("log", vec![target]),
                    pool.func("log", vec![base]),
                    pool,
                );
                invert_for(exp, y, t, pool)
            } else {
                None
            }
        }
        ExprData::Func { name, args } if args.len() == 1 => {
            let inverse = match name.as_str() {
                "exp" => "log",
                "log" => "exp",
                "sin" => "asin",
                "asin" => "sin",
                "cos" => "acos",
                "acos" => "cos",
                "tan" => "atan",
                "atan" => "tan",
                "sinh" => "asinh",
                "asinh" => "sinh",
                "cosh" => "acosh",
                "acosh" => "cosh",
                "tanh" => "atanh",
                "atanh" => "tanh",
                "sqrt" => {
                    return invert_for(args[0], y, pool.pow(target, pool.integer(2_i32)), pool)
                }
                _ => return None,
            };
            invert_for(args[0], y, pool.func(inverse, vec![target]), pool)
        }
        _ => None,
    }
}

/// Verify an implicit general solution `G(x, y) = 0` of a second-order
/// equation by implicit differentiation: `y' = −G_x/G_y` and
/// `y'' = (∂ₓ + y'∂_y) y'` are substituted into the equation, and the residual
/// — a function of `(x, y)` and the constants `G_x`, `G_y` still carry — must
/// vanish. Through every point `(x, y)`, for every value of the other
/// constants, passes a member of the family (the constant `G` is affine in is
/// fixed by the point), so the residual has to vanish identically. It is
/// checked symbolically, then on a grid of sample points: a finite non-zero
/// sample refuses, and at least four agreeing samples certify.
pub(super) fn second_order_implicit_is_zero(
    input: &OdeInput,
    relation: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    let refuse = |m: &str| Err(DsolveError::VerificationFailed(m.to_string()));
    if input.order() != 2 {
        return refuse("second-order implicit verification on a non-second-order equation");
    }
    if !contains(relation, input.y, pool) {
        return refuse("implicit relation does not depend on y");
    }
    if !constants.iter().any(|&c| contains(relation, c, pool)) {
        return refuse("implicit relation carries no integration constant");
    }
    let gx = ddx(relation, input.x, pool)?;
    let gy = ddx(relation, input.y, pool)?;
    if super::is_zero(gy, pool) {
        return refuse("∂G/∂y vanishes identically");
    }
    let slope = simp(
        div(pool.mul(vec![pool.integer(-1_i32), gx]), gy, pool),
        pool,
    );
    let curv = simp(
        pool.add(vec![
            ddx(slope, input.x, pool)?,
            pool.mul(vec![slope, ddx(slope, input.y, pool)?]),
        ]),
        pool,
    );
    let residual = simp(
        subs1(
            subs1(input.equation, input.derivs[1], curv, pool),
            input.derivs[0],
            slope,
            pool,
        ),
        pool,
    );
    if super::is_zero(residual, pool) {
        return Ok(());
    }
    // Free symbols of the residual: x, y and the constants it still carries.
    let mut free = Vec::new();
    crate::eval::symbols::collect_free_symbols(residual, pool, &mut free);
    let allowed: Vec<ExprId> = [input.x, input.y]
        .into_iter()
        .chain(constants.iter().copied())
        .collect();
    if free.iter().any(|s| !allowed.contains(s)) {
        return refuse("the implicit residual mentions a symbol that is not x, y or a constant");
    }
    const XS: [f64; 3] = [0.31, 0.77, 1.43];
    const YS: [f64; 4] = [0.37, 0.83, 1.4, 2.6];
    const CS: [f64; 3] = [1.7, 0.6, 2.3];
    let mut agree = 0usize;
    for (i, &xv) in XS.iter().enumerate() {
        for &yv in &YS {
            let mut env = std::collections::HashMap::new();
            env.insert(input.x, xv);
            env.insert(input.y, yv);
            for (j, &c) in constants.iter().enumerate() {
                env.insert(c, CS[(i + j) % CS.len()] + 0.11 * j as f64);
            }
            let Some(v) = super::verify::eval(residual, &env, pool) else {
                continue;
            };
            let s = super::verify::eval(slope, &env, pool).unwrap_or(f64::NAN);
            let k = super::verify::eval(curv, &env, pool).unwrap_or(f64::NAN);
            if !v.is_finite() || !s.is_finite() || !k.is_finite() {
                continue;
            }
            let scale = 1.0 + s.abs() + k.abs() + yv.abs();
            if v.abs() > 1e-8 * scale {
                return Err(DsolveError::VerificationFailed(format!(
                    "implicit residual is {v:e} at x = {xv}, y = {yv}: {}",
                    pool.display(residual)
                )));
            }
            agree += 1;
        }
    }
    if agree >= 4 {
        Ok(())
    } else {
        refuse("too few implicit residual samples resolved to certify the relation")
    }
}
