//! Rational-function integration via Rothstein–Trager (Risch Gap 3).
//!
//! Computes the **logarithmic part** of `∫ A(x)/D(x) dx` for a rational function
//! over ℚ.  The classical pipeline is:
//!
//! 1. **Polynomial division** `A = Q·D + R` (deg R < deg D): `∫Q` is elementary.
//! 2. **Rothstein–Trager** on the proper part `R/D` with `D` squarefree, monic,
//!    `gcd(R, D) = 1`:
//!    ```text
//!      R(t) = res_x(R − t·D′, D),
//!      ∫ R/D dx = Σ_{c : R(c)=0}  c · log( gcd_x(R − c·D′, D) ).
//!    ```
//!    The roots `c` of the **resultant** are the residues; each contributes one
//!    logarithm whose argument is a polynomial GCD.
//!
//! ## Scope of this implementation
//!
//! - Handles a **squarefree** denominator with **rational** resultant roots
//!   (so the answer is a ℚ-linear combination of `log`s of ℚ-polynomials).
//! - Returns `None` (so the caller falls back) when:
//!   * the denominator is **not squarefree** — Hermite reduction is not yet wired
//!     in, so repeated factors are left to the rule engine; or
//!   * the resultant has **non-rational roots** — e.g. `1/(x²+1)` (residues `±i/2`),
//!     whose antiderivative needs `arctan`/algebraic-number logs not yet emitted.
//!
//! References: Rothstein (1976); Trager (1976); Bronstein (2005) §2.2–2.5;
//! SymPy `sympy/integrals/risch.py` (`residue_reduce`, `log_part`).

use rug::{Integer, Rational};

use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;

use super::poly_rde::{
    degree, poly_deriv, poly_integrate, poly_scale, qpoly_to_expr, rational_to_expr, trim, QPoly,
};
use super::rational_rde::{
    expr_to_qrational, poly_div_exact, poly_divrem, poly_gcd, poly_monic, poly_sub,
};

/// Attempt to integrate `expr` as a rational function of `var`.
///
/// Returns `Some(F)` with the antiderivative (no constant of integration) when
/// the Rothstein–Trager path succeeds, or `None` when `expr` is not a rational
/// function or falls outside the supported subset (see module docs), so the
/// caller can fall back to its existing behaviour.
pub fn try_integrate_rational(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (a, d) = expr_to_qrational(expr, var, pool)?;
    let a = trim(a);
    let d = trim(d);
    if d.is_empty() {
        return None; // division by zero — malformed
    }
    if degree(&d) < 1 {
        return None; // polynomial integrand — leave to the rule engine
    }

    // Normalise: make D monic and reduce A/D to lowest terms.
    let lc_inv = Rational::from(1) / d[degree(&d) as usize].clone();
    let d = poly_scale(&d, &lc_inv);
    let a = poly_scale(&a, &lc_inv);
    let g = poly_gcd(&a, &d);
    let a = poly_div_exact(&a, &g);
    let d = poly_monic(&poly_div_exact(&d, &g));
    if degree(&d) < 1 {
        return None; // reduced to a polynomial
    }

    // Polynomial part: A = Q·D + R.
    let (q, r) = poly_divrem(&a, &d);
    let poly_int = poly_integrate(&q);
    let r = trim(r);

    let mut terms: Vec<ExprId> = Vec::new();
    if !trim(poly_int.clone()).is_empty() {
        terms.push(qpoly_to_expr(&poly_int, var, pool));
    }

    if !r.is_empty() {
        // The proper part R/D needs a squarefree D for this Rothstein–Trager path.
        let dprime = poly_deriv(&d);
        if degree(&poly_gcd(&d, &dprime)) > 0 {
            return None; // not squarefree → Hermite reduction (future work)
        }
        let logs = rothstein_trager(&r, &d, &dprime, var, pool)?;
        terms.extend(logs);
    }

    Some(match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    })
}

/// Rothstein–Trager logarithmic part of `∫ R/D dx` (`D` squarefree, monic,
/// `gcd(R, D) = 1`).  Returns `None` if any resultant root is non-rational.
fn rothstein_trager(
    r: &QPoly,
    d: &QPoly,
    dprime: &QPoly,
    var: ExprId,
    pool: &ExprPool,
) -> Option<Vec<ExprId>> {
    // Build R(t) = res_x(R − t·D′, D) over ℤ.  Clear denominators of R and D′ by a
    // *single* common factor L so the resultant's roots are exactly the residues.
    let l = lcm_denoms(&[r, dprime]);
    let r_int = poly_scale(r, &Rational::from(l.clone()));
    let dp_int = poly_scale(dprime, &Rational::from(l));
    let m = lcm_denoms(&[d]);
    let d_int = poly_scale(d, &Rational::from(m));

    let t = pool.symbol("$rt_param$", Domain::Complex);
    let r_expr = qpoly_to_expr(&r_int, var, pool);
    let dp_expr = qpoly_to_expr(&dp_int, var, pool);
    // P(x, t) = R(x) − t·D′(x)
    let p_expr = pool.add(vec![
        r_expr,
        pool.mul(vec![pool.integer(-1_i32), t, dp_expr]),
    ]);
    let d_expr = qpoly_to_expr(&d_int, var, pool);

    let res = crate::poly::resultant(p_expr, d_expr, var, pool).ok()?;
    let rt_poly = UniPoly::from_symbolic(res.value, t, pool).ok()?;
    if rt_poly.degree() < 1 {
        return None;
    }
    let fac = rt_poly.factor_z().ok()?;

    // Collect distinct rational roots (residues); bail on any non-linear factor.
    let mut residues: Vec<Rational> = Vec::new();
    for (f, _mult) in &fac.factors {
        if f.degree() != 1 {
            return None; // non-rational residue — needs algebraic extension / arctan
        }
        let coeffs = f.coefficients(); // ascending: [b, a] for a·t + b
        let b = Rational::from(coeffs[0].clone());
        let a = Rational::from(coeffs[1].clone());
        let c = -b / a;
        if !residues.contains(&c) {
            residues.push(c);
        }
    }

    // Each residue c contributes c·log(gcd_x(R − c·D′, D)).
    let mut logs: Vec<ExprId> = Vec::new();
    let mut covered = 0i64;
    for c in &residues {
        let shifted = poly_sub(r, &poly_scale(dprime, c));
        let gc = poly_monic(&poly_gcd(&shifted, d));
        if degree(&gc) <= 0 {
            continue;
        }
        covered += degree(&gc);
        let arg = qpoly_to_expr(&gc, var, pool);
        let log_arg = pool.func("log", vec![arg]);
        let term = if *c == 1 {
            log_arg
        } else {
            pool.mul(vec![rational_to_expr(c, pool), log_arg])
        };
        logs.push(term);
    }

    // Soundness guard: every root of D must be accounted for by some residue.
    if covered != degree(d) {
        return None;
    }
    Some(logs)
}

/// LCM of all coefficient denominators across the given polynomials.
fn lcm_denoms(polys: &[&QPoly]) -> Integer {
    let mut l = Integer::from(1);
    for p in polys {
        for c in p.iter() {
            l = l.lcm(c.denom());
        }
    }
    l
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprData;

    fn pool() -> ExprPool {
        ExprPool::new()
    }

    /// Numeric evaluator for verification (Integer/Rational/Add/Mul/Pow/log).
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
            ExprData::Func { ref name, ref args } if name == "log" && args.len() == 1 => {
                eval(args[0], x, xv, pool).ln()
            }
            other => panic!("eval: unsupported {other:?}"),
        }
    }

    /// Verify d/dx F = integrand numerically at several points.
    fn verify(expr: ExprId, antideriv: ExprId, x: ExprId, pool: &ExprPool) {
        let d = crate::diff::diff(antideriv, x, pool).unwrap();
        let ds = crate::simplify::engine::simplify(d.value, pool).value;
        for &xv in &[1.7_f64, 2.3, 3.9] {
            let lhs = eval(ds, x, xv, pool);
            let rhs = eval(expr, x, xv, pool);
            assert!(
                (lhs - rhs).abs() < 1e-7,
                "d/dx F ≠ f at x={xv}: {lhs} vs {rhs} (F={})",
                pool.display(antideriv)
            );
        }
    }

    #[test]
    fn one_over_x2_minus_1() {
        // ∫ 1/(x²−1) dx = ½ log(x−1) − ½ log(x+1).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(-1), Rational::from(0), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.mul(vec![
            pool.integer(1_i32),
            pool.pow(den, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn two_x_over_x2_plus_1() {
        // ∫ 2x/(x²+1) dx = log(x²+1).  (Residue c = 1, rational.)
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let x2p1 = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.mul(vec![
            pool.integer(2_i32),
            x,
            pool.pow(x2p1, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn one_over_x2_minus_x() {
        // ∫ 1/(x²−x) dx = ∫ 1/(x(x−1)) dx = −log(x) + log(x−1).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(0), Rational::from(-1), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.pow(den, pool.integer(-1_i32));
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn improper_with_log_part() {
        // ∫ x³/(x²−1) dx = x²/2 + ½ log(x²−1)  (improper: polynomial + log part).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = poly_to_expr(
            &[Rational::from(-1), Rational::from(0), Rational::from(1)],
            x,
            &pool,
        );
        let f = pool.mul(vec![
            pool.pow(x, pool.integer(3_i32)),
            pool.pow(den, pool.integer(-1_i32)),
        ]);
        let result = try_integrate_rational(f, x, &pool).expect("rational integral");
        verify(f, result, x, &pool);
    }

    #[test]
    fn complex_residues_unsupported() {
        // ∫ 1/(x²+1) dx needs arctan (residues ±i/2) — not handled here → None.
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let den = pool.add(vec![pool.pow(x, pool.integer(2_i32)), pool.integer(1_i32)]);
        let f = pool.pow(den, pool.integer(-1_i32));
        assert!(try_integrate_rational(f, x, &pool).is_none());
    }

    #[test]
    fn non_squarefree_unsupported() {
        // ∫ 1/(x+1)² dx = −1/(x+1): squarefree-only RT path declines (Hermite needed).
        let pool = pool();
        let x = pool.symbol("x", Domain::Real);
        let xp1 = pool.add(vec![x, pool.integer(1_i32)]);
        let f = pool.pow(xp1, pool.integer(-2_i32));
        assert!(try_integrate_rational(f, x, &pool).is_none());
    }

    fn poly_to_expr(coeffs: &[Rational], x: ExprId, pool: &ExprPool) -> ExprId {
        qpoly_to_expr(&coeffs.to_vec(), x, pool)
    }
}
