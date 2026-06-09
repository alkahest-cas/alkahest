//! `∫ R(x, α)·exp(β) dx` where `β` is an **algebraic** function of `x` — the
//! non-diagonal-`f` exp case (Risch milestone **M1**, PR2).
//!
//! ## The math
//!
//! For `∫ R·exp(β) dx` with `β` an algebraic function of `x`, seek an
//! antiderivative of the form `v·exp(β)`.  Then
//!
//! ```text
//!   D(v·exp(β)) = (D(v) + D(β)·v)·exp(β),
//! ```
//!
//! so `v` must satisfy the Risch differential equation
//!
//! ```text
//!   D(v) + f·v = R     with     f = D(β),
//! ```
//!
//! all inside the algebraic function field `ℚ(x)(α)` where `α` generates `β`.
//! When `β` is algebraic, `f = D(β)` is a **non-base** element of `ℚ(x)(α)` (it
//! carries `α`-powers) — exactly the case that `solve_alg_rde_general` (PR1,
//! in [`super::alg_rde`]) handles.
//!
//! ## Reachable shape (kept narrow & sound)
//!
//! - `β` is a single algebraic generator that is a **radical**
//!   `α = p(x)^{1/n}` (`p ∈ ℚ[x]`, `n ≥ 2`), possibly scaled by a rational
//!   constant `c` (`exp(c·α)`, so `f = c·D(α)`).  Examples: `exp(√x)`,
//!   `exp(√(x+1))`, `exp(∛x)`, `exp(2√x)`.
//! - The remaining factor `R` (after dividing out the single `exp(β)`) is a
//!   rational function in `x` and `α` — i.e. it parses into an `AlgElem` over
//!   `ℚ(x)[y]/(yⁿ − p)`.
//!
//! Anything outside this shape (multiple `exp` factors, `β` transcendental or
//! nested, `R` not expressible in the field) → `None`, so the caller falls
//! through to the ordinary dispatch.  A computed antiderivative is emitted only
//! after the numeric soundness gate `d/dx F = integrand` passes; a failed gate
//! or an unsolved RDE yields `None`/`NonElementary`, never a wrong answer.
//!
//! ## Worked example
//!
//! `∫ exp(√x)·(1/(2√x) + 1/2) dx = √x·exp(√x)`.  Field `ℚ(x)(α)`, `α² = x`,
//! `f = D(α) = (1/(2x))·α`.  `R = 1/(2√x) + 1/2 = (1/(2x))·α + 1/2`.  Solving
//! `D(v) + f·v = R` gives `v = α = √x`, so `F = √x·exp(√x)` — verified.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::IntegrationError;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::alg_field::AlgExtension;
use super::alg_rde::solve_alg_rde_general;
use super::exp_case::{build_rational, decompose_over_alg_generator, detect_radical_generator};
use super::poly_rde::{contains_subexpr, is_free_of_var, qpoly_to_expr};
use super::tower::find_generators;

/// Public entry: try to integrate `∫ R(x, α)·exp(β) dx` where `β` is an
/// algebraic function of `var` given by a single radical `α = p(x)^{1/n}`.
///
/// Returns `None` when the integrand is not of this shape (the caller falls
/// through to the ordinary dispatch).  When it is, returns `Some(Ok(F))` with a
/// numerically-verified antiderivative, or `Some(Err(NonElementary))` when the
/// in-field Risch DE has no solution within the ansatz bounds.
pub fn try_integrate_exp_of_algebraic(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    // 1. Find the single exp generator exp(β) whose argument β is an *algebraic*
    //    function of var (it carries a radical of a polynomial in var, so it is
    //    neither free-of-var nor itself a transcendental-tower generator).
    let exp_gen = detect_exp_of_algebraic(expr, var, pool)?;
    let beta = match pool.get(exp_gen) {
        ExprData::Func { ref args, .. } => args[0],
        _ => return None,
    };

    // 2. Build the algebraic field ℚ(x)(α) from the single radical α = p^{1/n}
    //    appearing in β.  β must contain exactly one radical generator.
    let (n, p) = detect_radical_generator(beta, var, pool)?;
    if n < 2 {
        return None;
    }
    let e = AlgExtension::radical(n, &p);

    // 3. Express β itself as an AlgElem over the field (β = c·α, or any element
    //    expressible in ℚ(x)(α)).  Then f = D(β) is the (generally non-base)
    //    twist of the Risch DE.
    let beta_elem = decompose_over_alg_generator(beta, n, &p, &e, var, pool)?;
    let f = e.derivation(&beta_elem);

    // 4. Divide out the single exp(β) factor and parse the remaining R into an
    //    AlgElem over ℚ(x)(α).  Distribute the division across additive terms so
    //    the exp(β)·exp(β)⁻¹ cancellation actually fires.
    let neg1 = pool.integer(-1_i32);
    let inv_exp = pool.pow(exp_gen, neg1);
    let div_term =
        |t: ExprId, pool: &ExprPool| -> ExprId { simplify(pool.mul(vec![t, inv_exp]), pool).value };
    let r_expr = match pool.get(expr) {
        ExprData::Add(args) => {
            let parts: Vec<ExprId> = args.iter().map(|&t| div_term(t, pool)).collect();
            simplify(pool.add(parts), pool).value
        }
        _ => div_term(expr, pool),
    };
    // R must no longer mention exp(β); otherwise the cancellation failed and this
    // is not the supported shape.
    if contains_subexpr(r_expr, exp_gen, pool) {
        return None;
    }
    let r_elem = decompose_over_alg_generator(r_expr, n, &p, &e, var, pool)?;

    // 5. Solve D(v) + f·v = R over ℚ(x)(α) (non-diagonal coupled solver, PR1).
    //    None ⇒ no antiderivative of the form v·exp(β): certify NonElementary
    //    (the field/ansatz is complete for this radical shape — mirroring how the
    //    other radical-RDE paths report their failures).
    let v = match solve_alg_rde_general(&e, &f, &r_elem) {
        Some(v) => v,
        None => {
            return Some(Err(IntegrationError::NonElementary(format!(
                "∫ {} dx: the Risch differential equation D(v)+D(β)·v=R over ℚ(x)(α) \
                 has no rational solution, so no antiderivative of the form v·exp(β) exists",
                pool.display(expr),
            ))));
        }
    };

    // 6. Reconstruct F = (Σ vᵢ·αⁱ)·exp(β).
    let p_expr = qpoly_to_expr(&p, var, pool);
    let mut v_terms: Vec<ExprId> = Vec::new();
    for (i, vi) in v.iter().enumerate() {
        if vi.numer().is_empty() {
            continue; // zero coefficient
        }
        let coeff = build_rational(vi.numer(), vi.denom(), var, pool);
        let term = if i == 0 {
            coeff
        } else {
            let yi = pool.pow(p_expr, pool.rational(i as i32, n as i32));
            pool.mul(vec![coeff, yi])
        };
        v_terms.push(term);
    }
    let v_expr = match v_terms.len() {
        0 => pool.integer(0_i32),
        1 => v_terms[0],
        _ => pool.add(v_terms),
    };
    let f_expr = simplify(pool.mul(vec![v_expr, exp_gen]), pool).value;

    // 7. Soundness gate: d/dx F must equal the integrand numerically.
    if !verify_derivative(f_expr, expr, var, pool) {
        return None;
    }

    let mut log = DerivationLog::new();
    log.push(RewriteStep::simple("risch_exp_of_algebraic", expr, f_expr));
    Some(Ok(DerivedExpr::with_log(f_expr, log)))
}

/// Find the single `exp(β)` generator whose argument `β` is an algebraic
/// function of `var`: it depends on `var`, has no exp/log tower generators of
/// its own, and contains a radical of a polynomial in `var`.  Returns `None`
/// unless there is exactly one such generator.
fn detect_exp_of_algebraic(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let mut found: Vec<ExprId> = Vec::new();
    scan_exp_of_algebraic(expr, var, pool, &mut found);
    found.dedup();
    if found.len() == 1 {
        Some(found.remove(0))
    } else {
        None
    }
}

fn scan_exp_of_algebraic(expr: ExprId, var: ExprId, pool: &ExprPool, out: &mut Vec<ExprId>) {
    match pool.get(expr) {
        ExprData::Func { ref name, ref args } if name == "exp" && args.len() == 1 => {
            let beta = args[0];
            if is_exp_arg_algebraic(beta, var, pool) && !out.contains(&expr) {
                out.push(expr);
            }
            // Do not recurse into β: a nested exp inside an algebraic β is out of
            // scope and would be caught by the "more than one generator" guard
            // elsewhere; recursing could double-count.
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            for &a in &args {
                scan_exp_of_algebraic(a, var, pool, out);
            }
        }
        ExprData::Pow { base, exp } => {
            scan_exp_of_algebraic(base, var, pool, out);
            scan_exp_of_algebraic(exp, var, pool, out);
        }
        _ => {}
    }
}

/// Is `beta` an algebraic (radical) function of `var` suitable as an exp
/// argument?  It must depend on `var`, contain no exp/log tower generators, and
/// contain exactly one radical generator `p(x)^{1/n}`.
fn is_exp_arg_algebraic(beta: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if is_free_of_var(beta, var, pool) {
        return false;
    }
    // No transcendental (exp/log) tower generators inside β.
    if !find_generators(beta, var, pool).is_empty() {
        return false;
    }
    detect_radical_generator(beta, var, pool).is_some()
}

// ---------------------------------------------------------------------------
// Numeric verification (sound gate)
// ---------------------------------------------------------------------------

/// Verify `d/dx f = integrand` numerically at several sample points where the
/// radical and exp are real.
fn verify_derivative(f: ExprId, integrand: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    let Ok(df) = crate::diff::diff(f, var, pool) else {
        return false;
    };
    let ds = simplify(df.value, pool).value;
    let mut checked = 0;
    for &xv in &[0.55_f64, 1.3, 2.1, 3.4] {
        let (Some(lhs), Some(rhs)) = (eval(ds, var, xv, pool), eval(integrand, var, xv, pool))
        else {
            continue;
        };
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        if (lhs - rhs).abs() > 1e-6 * (1.0 + rhs.abs()) {
            return false;
        }
        checked += 1;
    }
    checked >= 2
}

fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> Option<f64> {
    if expr == x {
        return Some(xv);
    }
    match pool.get(expr) {
        ExprData::Integer(n) => Some(n.0.to_f64()),
        ExprData::Rational(r) => Some(r.0.to_f64()),
        ExprData::Add(args) => args
            .iter()
            .try_fold(0.0, |s, &a| Some(s + eval(a, x, xv, pool)?)),
        ExprData::Mul(args) => args
            .iter()
            .try_fold(1.0, |s, &a| Some(s * eval(a, x, xv, pool)?)),
        ExprData::Pow { base, exp } => Some(eval(base, x, xv, pool)?.powf(eval(exp, x, xv, pool)?)),
        ExprData::Func { ref name, ref args } if args.len() == 1 => {
            let a = eval(args[0], x, xv, pool)?;
            match name.as_str() {
                "exp" => Some(a.exp()),
                "log" => Some(a.ln()),
                "sqrt" => Some(a.sqrt()),
                "cbrt" => Some(a.cbrt()),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    /// Numeric check that `d/dx F = integrand` at the given sample points.
    fn assert_diff_matches(f: ExprId, integrand: ExprId, x: ExprId, pool: &ExprPool, pts: &[f64]) {
        let ds = simplify(crate::diff::diff(f, x, pool).unwrap().value, pool).value;
        for &xv in pts {
            let lhs = eval(ds, x, xv, pool).unwrap();
            let rhs = eval(integrand, x, xv, pool).unwrap();
            assert!(
                (lhs - rhs).abs() < 1e-6 * (1.0 + rhs.abs()),
                "x={xv}: d/dx F = {lhs}, integrand = {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    /// Headline: ∫ exp(√x)·(1/(2√x) + 1/2) dx = √x·exp(√x).
    #[test]
    fn exp_sqrt_x_with_coeff_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_sqrt_x = pool.func("exp", vec![sqrt_x]);
        // coeff = 1/(2√x) + 1/2
        let half = pool.rational(1_i32, 2_i32);
        let inv_2sqrt = pool.mul(vec![half, pool.pow(sqrt_x, pool.integer(-1_i32))]);
        let coeff = pool.add(vec![inv_2sqrt, half]);
        let integrand = pool.mul(vec![exp_sqrt_x, coeff]);

        let res = try_integrate_exp_of_algebraic(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.6, 1.4, 2.7]);
    }

    /// Elementary bare exp: ∫ exp(√x) dx = 2(√x − 1)·exp(√x).
    #[test]
    fn exp_sqrt_x_bare_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let integrand = pool.func("exp", vec![sqrt_x]);

        let res = try_integrate_exp_of_algebraic(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.6, 1.4, 2.7]);
    }

    /// Non-elementary: ∫ exp(√x)/x dx (Ei-type) → NonElementary, never a wrong
    /// elementary form.
    #[test]
    fn exp_sqrt_x_over_x_nonelementary() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_sqrt_x = pool.func("exp", vec![sqrt_x]);
        let integrand = pool.mul(vec![exp_sqrt_x, pool.pow(x, pool.integer(-1_i32))]);

        let res = try_integrate_exp_of_algebraic(integrand, x, &pool);
        assert!(
            matches!(res, Some(Err(IntegrationError::NonElementary(_)))),
            "∫ exp(√x)/x dx must be NonElementary; got {res:?}"
        );
    }

    /// √(x+1) variant: ∫ exp(√(x+1))/(2√(x+1)) dx = exp(√(x+1)).
    #[test]
    fn exp_sqrt_x_plus_1_integrates() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let rad = pool.func("sqrt", vec![xp1]);
        let exp_rad = pool.func("exp", vec![rad]);
        let half = pool.rational(1_i32, 2_i32);
        let coeff = pool.mul(vec![half, pool.pow(rad, pool.integer(-1_i32))]);
        let integrand = pool.mul(vec![exp_rad, coeff]);

        let res = try_integrate_exp_of_algebraic(integrand, x, &pool)
            .expect("recognized")
            .expect("elementary");
        assert_diff_matches(res.value, integrand, x, &pool, &[0.6, 1.4, 2.7]);
    }

    /// Declines plain ∫ x·exp(x) dx (β = x is not algebraic).
    #[test]
    fn declines_plain_exp() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.mul(vec![x, pool.func("exp", vec![x])]);
        assert!(try_integrate_exp_of_algebraic(f, x, &pool).is_none());
    }

    /// Declines ∫ exp(x²) dx (β = x² is polynomial, not algebraic).
    #[test]
    fn declines_exp_poly() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let f = pool.func("exp", vec![x2]);
        assert!(try_integrate_exp_of_algebraic(f, x, &pool).is_none());
    }

    // ---- End-to-end through the public engine (routing reaches the hook) ----

    /// ∫ exp(√x)·(1/(2√x)+1/2) dx through the public `integrate` entry point.
    #[test]
    fn headline_end_to_end_public_engine() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_sqrt_x = pool.func("exp", vec![sqrt_x]);
        let half = pool.rational(1_i32, 2_i32);
        let inv_2sqrt = pool.mul(vec![half, pool.pow(sqrt_x, pool.integer(-1_i32))]);
        let coeff = pool.add(vec![inv_2sqrt, half]);
        let integrand = pool.mul(vec![exp_sqrt_x, coeff]);

        // Routing must classify this as a Risch form.
        assert!(super::super::contains_risch_form(integrand, x, &pool));

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "should integrate; got {res:?}");
        assert_diff_matches(res.unwrap().value, integrand, x, &pool, &[0.6, 1.4, 2.7]);
    }

    /// ∫ exp(√x) dx = 2(√x−1)·exp(√x) through the public engine.
    #[test]
    fn bare_exp_sqrt_end_to_end_public_engine() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let integrand = pool.func("exp", vec![sqrt_x]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(res.is_ok(), "∫ exp(√x) dx should integrate; got {res:?}");
        assert_diff_matches(res.unwrap().value, integrand, x, &pool, &[0.6, 1.4, 2.7]);
    }

    /// ∫ exp(√x)/x dx → NonElementary through the public engine (no wrong form).
    #[test]
    fn nonelementary_end_to_end_public_engine() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let sqrt_x = pool.func("sqrt", vec![x]);
        let exp_sqrt_x = pool.func("exp", vec![sqrt_x]);
        let integrand = pool.mul(vec![exp_sqrt_x, pool.pow(x, pool.integer(-1_i32))]);

        let res = crate::integrate::engine::integrate(integrand, x, &pool);
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "∫ exp(√x)/x dx must be NonElementary; got {res:?}"
        );
    }
}
