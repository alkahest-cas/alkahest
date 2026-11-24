//! `∫ h(sin θ) dx` for an integrand that is a function of **one** `sin θ` (or
//! `cos θ`) alone, through a radical: `s = sin θ`, `dx = ds/(θ′·cos θ)`.
//!
//! `cos θ = ±√(1 − s²)`, and the sign is not a function of `s`, so the
//! substitution gives
//!
//! ```text
//! ∫ h(sin θ) dx = sign(cos θ)·G(sin θ),   G(s) = ∫ h(s)/(θ′·√(1 − s²)) ds
//! ```
//!
//! (`d/dx` of the right side is `sign(cos θ)·h·cos θ/|cos θ| = h`).  Odd powers
//! of `cos θ` are exactly what the generator substitution refuses, and they
//! are the shape of `∫√(1 + sin x)·log(sin x) dx` (Charlwood #41):
//! `√(1+s)/√(1−s²) = 1/√(1−s)` on the real domain and `G` is elementary.
//!
//! Only integrands that carry a radical of the generator are attempted (the
//! purely rational ones belong to Bioche and the half-angle routes), the inner
//! integral goes through the full integrator, and the result is gated against
//! the original integrand in `x`.

use std::collections::HashMap;

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::engine::{integrate, is_free_of, is_linear_in, verify_antiderivative_status};

pub(crate) const SINE_SUBST_RULE: &str = "int_sine_substitution_signed";

/// The single `sin θ` / `cos θ` node every occurrence of `var` sits under.
fn generator(e: ExprId, var: ExprId, pool: &ExprPool, g: &mut Option<ExprId>) -> bool {
    if is_free_of(e, var, pool) {
        return true;
    }
    match pool.get(e) {
        ExprData::Func { name, args }
            if (name == "sin" || name == "cos")
                && args.len() == 1
                && is_linear_in(args[0], var, pool).is_some() =>
        {
            match g {
                Some(prev) => *prev == e,
                None => {
                    *g = Some(e);
                    true
                }
            }
        }
        ExprData::Func { args, .. } | ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&a| generator(a, var, pool, g))
        }
        ExprData::Pow { base, exp } => {
            generator(base, var, pool, g) && generator(exp, var, pool, g)
        }
        _ => false,
    }
}

/// `true` when some non-integer power (or `sqrt`) depends on `var`.
fn has_radical(e: ExprId, var: ExprId, pool: &ExprPool) -> bool {
    if is_free_of(e, var, pool) {
        return false;
    }
    match pool.get(e) {
        ExprData::Pow { base, exp } => {
            matches!(pool.get(exp), ExprData::Rational(_)) || has_radical(base, var, pool)
        }
        ExprData::Func { name, args } => {
            name == "sqrt" || args.iter().any(|&a| has_radical(a, var, pool))
        }
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().any(|&a| has_radical(a, var, pool))
        }
        _ => false,
    }
}

pub(crate) fn try_sine_substitution(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<DerivedExpr<ExprId>> {
    if !has_radical(expr, var, pool) {
        return None;
    }
    let mut g = None;
    if !generator(expr, var, pool, &mut g) {
        return None;
    }
    let g = g?;
    let ExprData::Func { name, args } = pool.get(g) else {
        return None;
    };
    let theta = args[0];
    let (slope, _) = is_linear_in(theta, var, pool)?;
    let is_sin = name == "sin";
    let s = pool.symbol("__sine_subst_s", crate::kernel::Domain::Real);
    let h = crate::kernel::subs(expr, &HashMap::from([(g, s)]), pool);
    if !is_free_of(h, var, pool) {
        return None;
    }
    // ds = θ′·cos θ dx (sin) or −θ′·sin θ dx (cos); the other function is
    // ±√(1 − s²).
    let one_minus_s2 = pool.add(vec![
        pool.integer(1_i32),
        pool.mul(vec![pool.integer(-1_i32), pool.pow(s, pool.integer(2_i32))]),
    ]);
    let inner = simplify(
        pool.mul(vec![
            h,
            pool.pow(one_minus_s2, pool.rational(-1, 2)),
            pool.pow(slope, pool.integer(-1_i32)),
            pool.integer(if is_sin { 1_i32 } else { -1_i32 }),
        ]),
        pool,
    )
    .value;
    crate::budget::check().ok()?;
    let big_g = integrate(inner, s, pool).ok()?;
    let back = crate::kernel::subs(big_g.value, &HashMap::from([(s, g)]), pool);
    let other = pool.func(if is_sin { "cos" } else { "sin" }, vec![theta]);
    let signed = simplify(pool.mul(vec![pool.func("sign", vec![other]), back]), pool).value;
    for f in [signed, simplify(back, pool).value] {
        crate::budget::check().ok()?;
        if is_free_of(f, s, pool) && verify_antiderivative_status(f, expr, var, pool).is_some() {
            let mut log = DerivationLog::new();
            log.push(RewriteStep::simple(SINE_SUBST_RULE, expr, f));
            return Some(DerivedExpr::with_log(f, log.merge(big_g.log)));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    fn check(src: &str) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
        let r = try_sine_substitution(f, x, &pool).expect("closes");
        assert!(verify_antiderivative_status(r.value, f, x, &pool).is_some());
        // Real and right on both signs of cos x (x = 0.4 and x = 2.6).
        for at in [0.4, 1.1, 2.0, 2.6] {
            let mut env = HashMap::new();
            env.insert(x, at);
            let d = crate::diff::diff(r.value, x, &pool).unwrap().value;
            let a = crate::jit::eval_interp(d, &env, &pool).unwrap();
            let b = crate::jit::eval_interp(f, &env, &pool).unwrap();
            assert!(
                (a - b).abs() < 1e-8 * (1.0 + b.abs()),
                "{src} at {at}: {a} vs {b}"
            );
        }
    }

    /// Charlwood #41.
    #[test]
    fn charlwood_41() {
        check("sqrt(sin(x) + 1)*log(sin(x))");
    }

    #[test]
    fn plain_half_angle_radical() {
        check("sqrt(1 + sin(x))");
        check("sqrt(1 - cos(x))");
    }

    #[test]
    fn declines_without_a_radical_or_with_two_generators() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut syms = HashMap::from([("x".to_string(), x)]);
        for src in [
            "sin(x)/(1 + sin(x))",
            "sqrt(1 + sin(x))*cos(x)",
            "x*sqrt(sin(x))",
        ] {
            let f = crate::parse::parse(src, &pool, &mut syms).unwrap();
            assert!(try_sine_substitution(f, x, &pool).is_none(), "{src}");
        }
    }
}
