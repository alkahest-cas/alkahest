//! Lagrange (d'Alembert) equations `y = x·f(y') + g(y')`, solved in
//! parametric form.
//!
//! With `p = y'` as the parameter, differentiating `y = x f(p) + g(p)` in `x`
//! gives `p = f(p) + (x f'(p) + g'(p))·dp/dx`, i.e. the equation for `x(p)`
//!
//! `dx/dp = (f'(p)·x + g'(p)) / (p − f(p))`,
//!
//! which is *linear* in `x` and goes to the first-order cascade.  The general
//! solution is the curve `(x(p), y(p))`, `y(p) = x(p) f(p) + g(p)`; it can
//! rarely be written as `y(x)` (`y = 2xy' + y'²` needs the root of a cubic),
//! and it is not an implicit relation in the sense [`super::SolutionForm`]
//! means either, so it is reported through its own type.
//!
//! Every root `p₀` of `p − f(p)` gives a **singular solution**, the line
//! `y = x f(p₀) + g(p₀)` (its slope is `f(p₀) = p₀`); rational roots are found
//! and each line is verified by substitution.
//!
//! # Verification
//!
//! A parametric curve solves `F(x, y, y') = 0` exactly when, identically in
//! `p`,
//!
//! 1. `F(x(p), y(p), p) = 0` — the point lies on the equation with slope `p`;
//! 2. `y'(p) − p·x'(p) = 0` — and `p` really is the slope `dy/dx` of the curve.
//!
//! Both are checked: symbolically, or at sample values of `p` and the constant
//! where every resolved sample must vanish and at least four must resolve.
//! Clairaut's equation (`f(p) = p`) is the degenerate case `p − f ≡ 0` and is
//! left to the explicit Clairaut class.

use super::{
    contains, ddx, div, first_order, is_zero, simp, sub, subs1, ConstGen, DsolveError, OdeInput,
};
use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use std::collections::HashMap;

/// One parametric general-solution branch `x = x(p)`, `y = y(p)` of a
/// first-order equation, the parameter being the slope `p = y'`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ParametricBranch {
    /// The parameter symbol `p` (the slope `y'` along the curve).
    pub parameter: ExprId,
    /// `x` as a function of `p` and the constants.
    pub x_of_p: ExprId,
    /// `y` as a function of `p` and the constants.
    pub y_of_p: ExprId,
    /// The integration constants `C1, …` in `x_of_p` / `y_of_p`.
    pub constants: Vec<ExprId>,
    /// Short label of the solving method (`"lagrange_parametric"`).
    pub method: &'static str,
}

/// The result of [`dsolve_parametric`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ParametricReport {
    /// Verified parametric general-solution branches (never empty on `Ok`).
    pub branches: Vec<ParametricBranch>,
    /// Verified singular solutions `y(x)` (lines `y = x f(p₀) + g(p₀)`), not
    /// members of the general family.
    pub singular: Vec<ExprId>,
}

/// Solve a first-order equation of Lagrange (d'Alembert) type
/// `y = x·f(y') + g(y')` in parametric form.
///
/// With `p = y'`, differentiating gives the *linear* equation
/// `dx/dp = (f'(p)·x + g'(p))/(p − f(p))`, solved by the first-order cascade;
/// then `y(p) = x(p)·f(p) + g(p)`.  The curve is verified by
/// `F(x(p), y(p), p) = 0` and `y'(p) = p·x'(p)`, identically in `p`.  Each
/// rational root `p₀` of `p − f(p)` gives a singular line
/// `y = x·f(p₀) + g(p₀)`, verified by substitution.
///
/// # Errors
///
/// [`DsolveError::Unsupported`] when the equation is not of Lagrange type (or
/// is Clairaut's, `f(p) ≡ p`, which [`super::dsolve`] solves explicitly), or the
/// linear equation for `x(p)` does not close; [`DsolveError::VerificationFailed`]
/// when the candidate curve does not verify.
pub fn dsolve_parametric(
    input: &OdeInput,
    pool: &ExprPool,
) -> Result<ParametricReport, DsolveError> {
    let unsupported = |m: &str| Err(DsolveError::Unsupported(m.to_string()));
    if input.order() != 1 {
        return unsupported("parametric (Lagrange) solving needs a first-order equation");
    }
    let (x, y, yp) = (input.x, input.y, input.derivs[0]);
    let eq = input.equation;
    if !contains(eq, yp, pool) {
        return unsupported("the equation does not contain y'");
    }
    // F = A(p)·y + B(p)·x + G(p)
    let a = simp(ddx(eq, y, pool)?, pool);
    let b = simp(ddx(eq, x, pool)?, pool);
    if [a, b]
        .iter()
        .any(|&c| contains(c, x, pool) || contains(c, y, pool))
        || is_zero(a, pool)
    {
        return unsupported("the equation is not of the form y = x·f(y') + g(y')");
    }
    let g_part = sub(
        eq,
        simp(
            pool.add(vec![pool.mul(vec![a, y]), pool.mul(vec![b, x])]),
            pool,
        ),
        pool,
    );
    if contains(g_part, x, pool) || contains(g_part, y, pool) {
        return unsupported("the equation is not of the form y = x·f(y') + g(y')");
    }
    let minus = |e: ExprId| pool.mul(vec![pool.integer(-1_i32), e]);
    let mut gen = ConstGen::new(input, pool);
    let p = parameter_symbol(input, pool);
    let f = subs1(div(minus(b), a, pool), yp, p, pool);
    let g = subs1(div(minus(g_part), a, pool), yp, p, pool);
    let p_minus_f = sub(p, f, pool);
    if is_zero(p_minus_f, pool) {
        return unsupported("Clairaut's equation y = x·y' + g(y') (solved explicitly by dsolve)");
    }

    // dx/dp = (f'·x + g')/(p − f)
    let big_x = pool.symbol("__lagrange_x", Domain::Real);
    let (lin, xp) = OdeInput::first_order(p, big_x, pool);
    let rhs = div(
        pool.add(vec![
            pool.mul(vec![ddx(f, p, pool)?, big_x]),
            ddx(g, p, pool)?,
        ]),
        p_minus_f,
        pool,
    );
    let lin = lin.with_equation(sub(xp, rhs, pool));
    let branches = first_order::solve(&lin, &mut gen, pool)?;
    let mut out = Vec::new();
    for br in branches {
        let Some(x_of_p) = br.y_of_x() else {
            continue;
        };
        let y_of_p = simp(pool.add(vec![pool.mul(vec![x_of_p, f]), g]), pool);
        if parametric_is_verified(input, p, x_of_p, y_of_p, &br.constants, pool).is_ok() {
            out.push(ParametricBranch {
                parameter: p,
                x_of_p,
                y_of_p,
                constants: br.constants.clone(),
                method: "lagrange_parametric",
            });
        }
    }
    if out.is_empty() {
        return Err(DsolveError::VerificationFailed(
            "no parametric candidate for the Lagrange equation verified".to_string(),
        ));
    }
    // Singular solutions: lines at the rational roots of p − f(p).
    let mut singular = Vec::new();
    for p0 in rational_roots_of(p_minus_f, p, pool) {
        let line = simp(
            pool.add(vec![
                pool.mul(vec![x, subs1(f, p, p0, pool)]),
                subs1(g, p, p0, pool),
            ]),
            pool,
        );
        if super::residual_is_zero(input, line, &[], pool).is_ok() && !singular.contains(&line) {
            singular.push(line);
        }
    }
    Ok(ParametricReport {
        branches: out,
        singular,
    })
}

/// `p`, or a variant of it not already used in the equation.
fn parameter_symbol(input: &OdeInput, pool: &ExprPool) -> ExprId {
    let mut used = std::collections::HashSet::new();
    super::collect_symbol_names(input.equation, pool, &mut used);
    for &s in [input.x, input.y].iter().chain(input.derivs.iter()) {
        if let ExprData::Symbol { name, .. } = pool.get(s) {
            used.insert(name);
        }
    }
    let mut name = "p".to_string();
    while used.contains(&name) {
        name.push('_');
    }
    pool.symbol(name, Domain::Real)
}

/// Rational roots of `e` (a rational function of `p` with rational
/// coefficients), or none.
fn rational_roots_of(e: ExprId, p: ExprId, pool: &ExprPool) -> Vec<ExprId> {
    super::special_quad::rational_zeros(simp(e, pool), p, pool)
}

/// Check the two conditions in the module docs for the curve
/// `(x(p), y(p))`.
pub(crate) fn parametric_is_verified(
    input: &OdeInput,
    p: ExprId,
    x_of_p: ExprId,
    y_of_p: ExprId,
    constants: &[ExprId],
    pool: &ExprPool,
) -> Result<(), DsolveError> {
    let mut m = HashMap::new();
    m.insert(input.x, x_of_p);
    m.insert(input.y, y_of_p);
    m.insert(input.derivs[0], p);
    let on_eq = simp(crate::kernel::subs::subs(input.equation, &m, pool), pool);
    let slope = simp(
        pool.add(vec![
            ddx(y_of_p, p, pool)?,
            pool.mul(vec![pool.integer(-1_i32), p, ddx(x_of_p, p, pool)?]),
        ]),
        pool,
    );
    for (what, r) in [("F(x(p), y(p), p)", on_eq), ("y'(p) − p·x'(p)", slope)] {
        if is_zero(r, pool) || is_zero(super::simp_plain(r, pool), pool) {
            continue;
        }
        let mut agree = 0usize;
        for &pv in &[0.37, 0.81, 1.3, 2.2, -0.6, -1.7] {
            for cs in super::verify::CONST_SETS.iter() {
                let mut env = HashMap::new();
                env.insert(p, pv);
                for (i, &c) in constants.iter().enumerate() {
                    env.insert(c, cs[i % cs.len()]);
                }
                let Some(v) = super::verify::eval(r, &env, pool) else {
                    return Err(DsolveError::VerificationFailed(format!(
                        "{what} is not evaluable: {}",
                        pool.display(r)
                    )));
                };
                if !v.is_finite() {
                    continue;
                }
                let scale = 1.0
                    + super::verify::eval(x_of_p, &env, pool).map_or(0.0, f64::abs)
                    + super::verify::eval(y_of_p, &env, pool).map_or(0.0, f64::abs);
                if v.abs() > 1e-9 * scale {
                    return Err(DsolveError::VerificationFailed(format!(
                        "{what} = {v:e} at p = {pv}: {}",
                        pool.display(r)
                    )));
                }
                agree += 1;
            }
        }
        if agree < 4 {
            return Err(DsolveError::VerificationFailed(format!(
                "too few samples of {what} resolved"
            )));
        }
    }
    Ok(())
}
