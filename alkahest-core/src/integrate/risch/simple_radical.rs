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
//! Over the radical integral basis `wₖ = zᵏ/dₖ` (eq 11) the derivation is
//! diagonal, `D(wₖ) = ωₖ·wₖ` (eq 22), so writing the integrand and a candidate
//! antiderivative in that basis gives `D(Σₖ vₖ wₖ) = Σₖ (vₖ' + ωₖ vₖ) wₖ` and the
//! integral **decouples per power `k`**:
//!
//! - `k = 0` (eq 23): `v₀' = A₀` — an ordinary `∫ A₀ dx` over `ℚ(x)`, handed to
//!   the rational integrator (this is where any new logarithms appear).
//! - `k ≥ 1` (eq 24): `vₖ' + ωₖ vₖ = Aₖ` — a Risch differential equation over
//!   `ℚ(x)`, solved by [`solve_rational_rde_generalized`].  No rational
//!   solution ⇒ the integral is **non-elementary**.
//!
//! Two cases of the basis:
//! - **Squarefree radicand `p`**: the basis collapses to the power basis
//!   `{1, y, …, y^{n−1}}` (`dₖ = 1`, `F = 1`), `ωₖ = (k/n)·p'/p`
//!   ([`try_integrate_simple_radical`]).
//! - **Non-squarefree radicand**: the explicit basis with `z = y/F`,
//!   `dₖ = ∏ⱼ Hⱼ^⌊kj/n⌋`, handled by `integrate_general_radical` (monic
//!   radicand; e.g. `∫∛(x²) = ⅗x^{5/3}`).
//!
//! ## Scope (what this slice does *not* do)
//!
//! - **Non-monic non-squarefree radicand** (e.g. `∛(4x²)`): the leading
//!   coefficient introduces an irrational constant `lc^{1/n}` — returns `None`.
//! - **Mixed algebraic + transcendental** (`∛x·exp(x)`, `∛(x+eˣ)`): the
//!   transcendental tower / mutual recursion is MD; such integrands are routed
//!   to the Risch engine, not here.
//! - **Genuine new-logarithm / torsion logic** beyond what the `k = 0` rational
//!   integration yields (the MC logarithmic part) is not attempted.

use crate::deriv::log::{DerivationLog, DerivedExpr, RewriteStep};
use crate::integrate::engine::{integrate_raw, IntegrationError};
use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::alg_field::AlgExtension;
use super::exp_case::{build_rational, decompose_over_alg_generator, detect_radical_generator};
use super::poly_rde::{
    degree, poly_deriv, poly_mul, poly_one, poly_scale, qpoly_to_expr, trim, QPoly,
};
use super::rational_integrate::yun;
use super::rational_rde::{poly_gcd, poly_pow, poly_sub, solve_rational_rde_generalized};

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
    // `D_{i−1} = 1`).  Non-squarefree radicand → general integral basis below.
    let p_prime = poly_deriv(&p);
    if degree(&poly_gcd(&p, &p_prime)) >= 1 {
        return integrate_general_radical(expr, n, &p, var, pool);
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

/// MA Step 1 for a **non-squarefree** radicand: the explicit radical integral
/// basis `wₖ = zᵏ / dₖ` (eq 11) with `z = y/F`, `dₖ = ∏ⱼ Hⱼ^⌊kj/n⌋`, and basis
/// derivative `ωₖ = (k/n)·H'/H − dₖ'/dₖ` (eq 22).  Decoupled per power `k`:
/// `vₖ' + ωₖ vₖ = Aₖ` where `Aₖ = cₖ·Fᵏ·dₖ` and `cₖ` is the `yᵏ`-coefficient of
/// the integrand.  Returns `None` for a non-monic radicand (the leading
/// coefficient would introduce an irrational constant `lc^{1/n}` — deferred).
fn integrate_general_radical(
    expr: ExprId,
    n: usize,
    a: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Result<DerivedExpr<ExprId>, IntegrationError>> {
    // Require a monic radicand (see doc-comment).
    if a.last().map(|c| *c != 1).unwrap_or(true) {
        return None;
    }

    // Squarefree-factor a = ∏ᵢ Aᵢ^i, then split each multiplicity i = n·qᵢ + rᵢ
    // into F = ∏ Aᵢ^{qᵢ} (pulled-out n-th powers) and H = ∏ Aᵢ^{rᵢ} (z = H^{1/n}).
    let a_factors = yun(a)?;
    let mut big_f = poly_one();
    let mut big_h = poly_one();
    for (ai, i) in &a_factors {
        let q = i / n;
        let r = i % n;
        if q > 0 {
            big_f = poly_mul(&big_f, &poly_pow(ai, q as u32));
        }
        if r > 0 {
            big_h = poly_mul(&big_h, &poly_pow(ai, r as u32));
        }
    }
    // Squarefree factors of H, for the basis denominators dₖ = ∏ⱼ Hⱼ^⌊kj/n⌋.
    let h_factors = yun(&big_h)?;
    let dk = |k: usize| -> QPoly {
        let mut d = poly_one();
        for (hj, j) in &h_factors {
            let e = (k * j) / n;
            if e > 0 {
                d = poly_mul(&d, &poly_pow(hj, e as u32));
            }
        }
        d
    };

    let e = AlgExtension::radical(n, a);
    let elem = decompose_over_alg_generator(expr, n, a, &e, var, pool)?;

    let h_prime = poly_deriv(&big_h);
    let a_expr = qpoly_to_expr(a, var, pool);
    let mut log = DerivationLog::new();
    let mut terms: Vec<ExprId> = Vec::new();

    for k in 0..n {
        // cₖ = numₖ/denₖ : the yᵏ-coefficient of the integrand.
        let (c_num, c_den) = match elem.get(k) {
            Some(r) => (r.numer().clone(), r.denom().clone()),
            None => (QPoly::new(), poly_one()),
        };
        if trim(c_num.clone()).is_empty() {
            continue;
        }
        let d_k = dk(k);
        let f_pow_k = poly_pow(&big_f, k as u32);

        // Aₖ = cₖ · Fᵏ · dₖ  (as a rational function num/den).
        let a_num = poly_mul(&poly_mul(&c_num, &f_pow_k), &d_k);
        let a_den = c_den;

        let v = if k == 0 {
            // ω₀ = 0 ⇒ ∫ A₀ dx over ℚ(x) (eq 23).  w₀ = 1, so add the integral.
            let a0 = build_rational(&a_num, &a_den, var, pool);
            match integrate_raw(a0, var, pool, &mut log) {
                Ok(v0) => terms.push(v0),
                Err(err) => return Some(Err(err)),
            }
            continue;
        } else {
            // ωₖ = (k·H'·dₖ − n·H·dₖ') / (n·H·dₖ)  (eq 22).
            let d_k_prime = poly_deriv(&d_k);
            let f_num = poly_sub(
                &poly_scale(&poly_mul(&h_prime, &d_k), &rug::Rational::from(k as i64)),
                &poly_scale(
                    &poly_mul(&big_h, &d_k_prime),
                    &rug::Rational::from(n as i64),
                ),
            );
            let f_den = poly_scale(&poly_mul(&big_h, &d_k), &rug::Rational::from(n as i64));
            match solve_rational_rde_generalized(&f_num, &f_den, &a_num, &a_den) {
                Some(sol) => sol,
                None => return Some(Err(non_elementary(expr, pool))),
            }
        };

        let (vn, vd) = v;
        if trim(vn.clone()).is_empty() {
            continue;
        }
        // wₖ = yᵏ / (Fᵏ·dₖ);  term = (vₖ / (Fᵏ·dₖ)) · yᵏ.
        let denom = poly_mul(&vd, &poly_mul(&f_pow_k, &d_k));
        let coeff = build_rational(&vn, &denom, var, pool);
        let yk = pool.pow(a_expr, pool.rational(k as i32, n as i32));
        terms.push(pool.mul(vec![coeff, yk]));
    }

    let raw = match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    };
    let simplified = simplify(raw, pool);
    log = log.merge(simplified.log);
    log.push(RewriteStep::simple(
        "simple_radical_risch_general",
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
    fn cbrt_func_routes_through_public_engine() {
        // `cbrt(x²)` (function form) reaches the algebraic engine via the
        // var-aware routing fix — previously the rule engine errored with
        // "∫ cbrt(non-trivial arg)".
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("cbrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let res = crate::integrate::engine::integrate(f, x, &pool)
            .expect("∫cbrt(x²) dx via public engine");
        let g = res.value;
        let h = 1e-6;
        let dnum = (eval(g, x, 1.4 + h, &pool) - eval(g, x, 1.4 - h, &pool)) / (2.0 * h);
        assert!(
            (dnum - 1.4_f64.powf(2.0 / 3.0)).abs() < 1e-4,
            "F = {}",
            pool.display(g)
        );
    }

    #[test]
    fn cbrt_of_constant_still_rule_engine() {
        // ∫ cbrt(5) dx = cbrt(5)·x — radicand is constant, must NOT route to the
        // algebraic engine (no regression from the var-aware routing).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let f = pool.func("cbrt", vec![pool.integer(5_i32)]);
        let res =
            crate::integrate::engine::integrate(f, x, &pool).expect("∫cbrt(5) dx = cbrt(5)·x");
        let g = res.value;
        // d/dx of result at x=2 should equal cbrt(5).
        let h = 1e-6;
        let dnum = (eval(g, x, 2.0 + h, &pool) - eval(g, x, 2.0 - h, &pool)) / (2.0 * h);
        assert!(
            (dnum - 5.0_f64.cbrt()).abs() < 1e-4,
            "F = {}",
            pool.display(g)
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
    fn integral_of_cbrt_x_squared_general_basis() {
        // ∫ ∛(x²) dx = ∫ x^{2/3} dx = (3/5) x^{5/3}.  Radicand x² is NOT
        // squarefree → exercises the general integral basis.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let cbrt_x2 = pool.func("cbrt", vec![pool.pow(x, pool.integer(2_i32))]);
        verify(cbrt_x2, x, &pool);
    }

    #[test]
    fn integral_using_basis_denominator() {
        // ∫ ∛(x²)² / x dx = ∫ x^{1/3} dx = (3/4) x^{4/3}.  Uses the basis element
        // w₂ = y²/x (nontrivial denominator d₂ = x).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let cbrt_x2 = pool.func("cbrt", vec![pool.pow(x, pool.integer(2_i32))]);
        let integrand = pool.mul(vec![
            pool.pow(cbrt_x2, pool.integer(2_i32)),
            pool.pow(x, pool.integer(-1_i32)),
        ]);
        verify(integrand, x, &pool);
    }

    #[test]
    fn integral_of_cbrt_linear_squared() {
        // ∫ ∛(x²+2x+1) dx = ∫ ∛((x+1)²) dx = (3/5)(x+1)^{5/3}; radicand
        // (x+1)² = x²+2x+1 is not squarefree.  (Built expanded — the engine
        // pre-expands via `simplify`; `expr_to_qpoly` reads expanded polys.)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let rad = pool.add(vec![
            pool.pow(x, pool.integer(2_i32)),
            pool.mul(vec![pool.integer(2_i32), x]),
            pool.integer(1_i32),
        ]);
        verify(pool.func("cbrt", vec![rad]), x, &pool);
    }

    #[test]
    fn non_monic_non_squarefree_returns_none() {
        // ∛((2x)²) = ∛(4x²): non-squarefree AND non-monic radicand (4x²) →
        // deferred (would introduce 4^{1/3}).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let two_x = pool.mul(vec![pool.integer(2_i32), x]);
        let rad = pool.pow(two_x, pool.integer(2_i32)); // 4x²
        let cbrt = pool.func("cbrt", vec![rad]);
        assert!(try_integrate_simple_radical(cbrt, x, &pool).is_none());
    }
}
