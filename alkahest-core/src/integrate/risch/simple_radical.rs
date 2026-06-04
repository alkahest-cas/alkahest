//! MA — simple-radical integral part (pure-algebraic case).
//!
//! Integrates `∫ R(x, y) dx` where `y = p(x)^{1/n}` is a **simple radical** of
//! degree `n ≥ 3` over `ℚ(x)` and `R` is a polynomial in `y` with rational-
//! function coefficients.  This is the pure-algebraic (`K(t) = ℚ(x)`, no
//! transcendental tower) slice of milestone **MA** in
//! `temp-alkahest/planning/risch.md` — the *integral part* (Steps 1–3 there),
//! built on the [`super::alg_field`] M0 substrate.
//!
//! ## Method (eqs 22–24)
//!
//! For a **squarefree** radicand `p`, the radical integral basis (eq 11)
//! collapses to the power basis `{1, y, …, y^{n−1}}` (all `D_{i−1} = 1`), and
//! `D(yʲ) = ωⱼ·yʲ` with `ωⱼ = (j/n)·p'/p` (eq 22).  Writing the integrand and a
//! candidate antiderivative in that basis, `D(Σⱼ vⱼ yʲ) = Σⱼ (vⱼ' + ωⱼ vⱼ) yʲ`,
//! so the integral **decouples per power `j`**:
//!
//! - `j = 0` (eq 23): `v₀' = b₀` — an ordinary `∫ b₀ dx` over `ℚ(x)`, handed to
//!   the rational integrator (this is where any new logarithms appear).
//! - `j ≥ 1` (eq 24): `vⱼ' + ωⱼ vⱼ = bⱼ` — a Risch differential equation over
//!   `ℚ(x)`, solved by [`solve_rational_rde_generalized`].  No rational
//!   solution ⇒ the integral is **non-elementary**.
//!
//! ## Scope (what this slice does *not* do)
//!
//! - **Non-squarefree radicand** (e.g. `∛(x²)`): needs the general van Hoeij
//!   integral basis (MB) — returns `None` (caller falls back).
//! - **Mixed algebraic + transcendental** (`∛x·exp(x)`, `∛(x+eˣ)`): the
//!   transcendental tower / mutual recursion is MD; such integrands are routed
//!   to the Risch engine, not here.
//! - **Genuine new-logarithm / torsion logic** beyond what the `j = 0` rational
//!   integration yields (the MC logarithmic part) is not attempted.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::{integrate_raw, IntegrationError};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::alg_field::AlgExtension;
use super::exp_case::{build_rational, decompose_over_alg_generator, detect_radical_generator};
use super::poly_rde::{degree, poly_deriv, poly_scale, qpoly_to_expr, trim, QPoly};
use super::rational_rde::poly_gcd;
use super::rational_rde::solve_rational_rde_generalized;

/// Attempt MA's integral part for a degree-`≥ 3` simple radical `y = p^{1/n}`
/// over `ℚ(x)` (pure-algebraic).
///
/// Returns:
/// - `None` — not applicable (no radical, degree `< 3`, non-squarefree radicand,
///   or the integrand is not a polynomial in `y` over `ℚ(x)`); the caller should
///   fall back to the existing algebraic engine.
/// - `Some(Err(NonElementary))` — a component Risch DE has no rational solution.
/// - `Some(Ok(F))` — the antiderivative `F`.
pub fn try_integrate_simple_radical(
    expr: ExprId,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    let (n, p) = detect_radical_generator(expr, var, pool)?;
    if n < 3 {
        return None; // degree 2 → existing genus-0 sqrt path
    }
    let p = trim(p);
    if degree(&p) < 1 {
        return None;
    }
    // Power-basis shortcut requires a squarefree radicand (eq 11 with all
    // `D_{i−1} = 1`).  Otherwise defer to the general integral basis (MB).
    let p_prime = poly_deriv(&p);
    if degree(&poly_gcd(&p, &p_prime)) >= 1 {
        return None;
    }

    let e = AlgExtension::radical(n, &p);
    let elem = decompose_over_alg_generator(expr, n, &p, &e, var, pool)?;

    let mut log = DerivationLog::new();
    let mut terms: Vec<ExprId> = Vec::new();

    for j in 0..n {
        // Coefficient `bⱼ = numⱼ/denⱼ` of `yʲ` in the integrand (0 if absent).
        let (num, den) = match elem.get(j) {
            Some(r) => (r.numer().clone(), r.denom().clone()),
            None => (QPoly::new(), vec![rug::Rational::from(1)]),
        };
        if trim(num.clone()).is_empty() {
            continue; // bⱼ = 0 ⇒ vⱼ = 0
        }

        if j == 0 {
            // eq 23: ∫ b₀ dx over ℚ(x) (the rational/logarithmic part).
            let b0 = build_rational(&num, &den, var, pool);
            match integrate_raw(b0, var, pool, &mut log) {
                Ok(v0) => terms.push(v0),
                Err(err) => return Some(Err(err)),
            }
        } else {
            // eq 24: vⱼ' + (j·p'/(n·p))·vⱼ = bⱼ.
            let f_num = poly_scale(&p_prime, &rug::Rational::from(j as i64));
            let f_den = poly_scale(&p, &rug::Rational::from(n as i64));
            let (vn, vd) = match solve_rational_rde_generalized(&f_num, &f_den, &num, &den) {
                Some(sol) => sol,
                None => return Some(Err(non_elementary(expr, pool))),
            };
            if trim(vn.clone()).is_empty() {
                continue;
            }
            let v_expr = build_rational(&vn, &vd, var, pool);
            // Multiply by yʲ = p^{j/n}.
            let p_expr = qpoly_to_expr(&p, var, pool);
            let yj = pool.pow(p_expr, pool.rational(j as i32, n as i32));
            terms.push(pool.mul(vec![v_expr, yj]));
        }
    }

    let raw = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    let simplified = simplify(raw, pool);
    log = log.merge(simplified.log);
    log.push(RewriteStep::simple(
        "simple_radical_risch",
        expr,
        simplified.value,
    ));
    Some(Ok(DerivedExpr::with_log(simplified.value, log)))
}

fn non_elementary(expr: ExprId, pool: &ExprPool) -> IntegrationError {
    IntegrationError::NonElementary(format!(
        "∫ {} dx: the Risch differential equation over ℚ(x) for a simple-radical \
         component has no rational solution",
        pool.display(expr),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprData};

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    /// Numeric evaluator supporting Add/Mul/Pow(rational)/Integer/Rational/var,
    /// `cbrt`, `sqrt`, `log` — enough for these antiderivatives and integrands.
    fn eval(expr: ExprId, x: ExprId, xv: f64, pool: &ExprPool) -> f64 {
        if expr == x {
            return xv;
        }
        match pool.get(expr) {
            ExprData::Integer(n) => n.0.to_f64(),
            ExprData::Rational(r) => r.0.to_f64(),
            ExprData::Add(args) => args.iter().map(|&a| eval(a, x, xv, pool)).sum(),
            ExprData::Mul(args) => args.iter().map(|&a| eval(a, x, xv, pool)).product(),
            ExprData::Pow { base, exp } => eval(base, x, xv, pool).powf(eval(exp, x, xv, pool)),
            ExprData::Func { ref name, ref args } if args.len() == 1 => {
                let a = eval(args[0], x, xv, pool);
                match name.as_str() {
                    "cbrt" => a.cbrt(),
                    "sqrt" => a.sqrt(),
                    "log" => a.ln(),
                    "exp" => a.exp(),
                    other => panic!("eval: unsupported func {other}"),
                }
            }
            other => panic!("eval: unsupported node {other:?}"),
        }
    }

    /// Integrate via the MA path, differentiate the result symbolically (now
    /// that `diff` handles fractional-power exponents), and assert
    /// `d/dx F = integrand` numerically.
    fn verify(integrand: ExprId, x: ExprId, pool: &ExprPool) {
        let res = try_integrate_simple_radical(integrand, x, pool)
            .expect("recognized as a simple radical")
            .expect("should be elementary");
        let f = res.value;
        let df = crate::diff::diff(f, x, pool).expect("F is differentiable");
        let ds = simplify(df.value, pool).value;
        for &xv in &[0.6_f64, 1.4, 2.7] {
            let lhs = eval(ds, x, xv, pool);
            let rhs = eval(integrand, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-9,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs}\n  F = {}",
                pool.display(f)
            );
        }
    }

    #[test]
    fn integral_of_cbrt_x() {
        // ∫ ∛x dx = (3/4) x^{4/3}.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        verify(pool.func("cbrt", vec![x]), x, &pool);
    }

    #[test]
    fn integral_of_x_times_cbrt_x() {
        // ∫ x·∛x dx = (3/7) x^{7/3}.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let integrand = pool.mul(vec![x, pool.func("cbrt", vec![x])]);
        verify(integrand, x, &pool);
    }

    #[test]
    fn integral_of_x_pow_two_thirds() {
        // ∫ x^{2/3} dx = (3/5) x^{5/3}.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let integrand = pool.pow(x, pool.rational(2_i32, 3_i32));
        verify(integrand, x, &pool);
    }

    #[test]
    fn integral_of_cbrt_x_over_x() {
        // ∫ ∛x / x dx = ∫ x^{-2/3} dx = 3 x^{1/3}.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let integrand = pool.mul(vec![
            pool.func("cbrt", vec![x]),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        verify(integrand, x, &pool);
    }

    #[test]
    fn fifth_root_integral() {
        // ∫ x^{2/5} dx = (5/7) x^{7/5}  (degree-5 radical, squarefree radicand).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let integrand = pool.pow(x, pool.rational(2_i32, 5_i32));
        verify(integrand, x, &pool);
    }

    #[test]
    fn cbrt_over_x2_plus_1_is_nonelementary() {
        // ∫ ∛x / (x²+1) dx is non-elementary.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2p1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let integrand = pool.mul(vec![
            pool.func("cbrt", vec![x]),
            pool.pow(x2p1, pool.integer(-1_i32)),
        ]);
        let res = try_integrate_simple_radical(integrand, x, &pool).expect("recognized");
        assert!(
            matches!(res, Err(IntegrationError::NonElementary(_))),
            "expected NonElementary, got {res:?}"
        );
    }

    #[test]
    fn routed_through_algebraic_engine() {
        // End-to-end: the pure-algebraic engine entry dispatches ∛x here.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let res =
            crate::integrate::algebraic::integrate_algebraic(pool.func("cbrt", vec![x]), x, &pool)
                .expect("∫∛x dx is elementary");
        let f = res.value;
        let h = 1e-6;
        let dnum = (eval(f, x, 1.4 + h, &pool) - eval(f, x, 1.4 - h, &pool)) / (2.0 * h);
        assert!(
            (dnum - 1.4_f64.cbrt()).abs() < 1e-4,
            "F = {}",
            pool.display(f)
        );
    }

    #[test]
    fn degree_two_returns_none() {
        // √x is degree 2 — left to the genus-0 sqrt engine (returns None here).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        assert!(try_integrate_simple_radical(pool.func("sqrt", vec![x]), x, &pool).is_none());
    }

    #[test]
    fn non_squarefree_radicand_returns_none() {
        // ∛(x²): radicand x² is not squarefree → deferred to MB (None).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2 = pool.pow(x, pool.integer(2_i32));
        let cbrt_x2 = pool.func("cbrt", vec![x2]);
        assert!(try_integrate_simple_radical(cbrt_x2, x, &pool).is_none());
    }
}
