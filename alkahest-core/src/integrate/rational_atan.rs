//! `∫ c·P′(t)/(P(t)² + k) dt = (c/√k)·atan(P(t)/√k)` for a polynomial `P`.
//!
//! The half-angle substitution turns Jeffrey's continuity examples —
//! `∫ (2 sin x + cos x + 1)/(cos²x − 2 sin x cos x + 2 sin x + 3) dx` and the
//! cosine-cubic pair — into rational functions such as
//! `(2t + 1)/(t⁴ + 2t³ + t² + 1)`, whose denominator is irreducible over ℚ with
//! residues `±i/2`.  The rational integrator's real form for such residues
//! goes through a `RootSum`, which the half-angle route cannot verify and so
//! suppresses, and the integral was declined.  But the denominator is
//! `(t² + t)² + 1` and the numerator is the derivative of `t² + t`: the answer
//! is `atan(t² + t)`.
//!
//! This module recognises exactly that shape: `D = λ·(P² + k)` with `k > 0`
//! (found by the polynomial square root of `D/λ` from the top) and
//! `A = c·P′`.  The caller verifies the result like every other candidate.

use rug::Rational;

use crate::kernel::{ExprId, ExprPool};
use crate::simplify::engine::simplify;

use super::risch::poly_rde::{degree, poly_deriv, poly_mul, qpoly_to_expr, trim, QPoly};
use super::risch::rational_rde::{expr_to_qrational, poly_divrem, poly_gcd, poly_sub};

/// `P` with `Q − P²` of degree `≤ 0`, for monic `Q` of even degree `2m ≥ 2`.
fn top_square_root(q: &QPoly) -> Option<QPoly> {
    let n = degree(q);
    if n < 2 || n % 2 != 0 {
        return None;
    }
    let m = (n / 2) as usize;
    let mut p = vec![Rational::from(0); m + 1];
    p[m] = Rational::from(1);
    // Coefficient of t^{2m−j} in P² is 2·p_m·p_{m−j} + Σ_{0<i<j} p_{m−i}·p_{m−j+i}.
    for j in 1..=m {
        let mut acc = Rational::from(0);
        for i in 1..j {
            acc += Rational::from(&p[m - i] * &p[m - j + i]);
        }
        let target = q.get(2 * m - j).cloned().unwrap_or_default();
        p[m - j] = (target - acc) / 2;
    }
    let rest = trim(poly_sub(q, &poly_mul(&p, &p)));
    (degree(&rest) <= 0).then_some(p)
}

/// `(c/√k)·atan(P/√k)` when `expr = A/D` matches the module's shape, else
/// `None`.  Unverified.
pub(crate) fn atan_of_polynomial(expr: ExprId, var: ExprId, pool: &ExprPool) -> Option<ExprId> {
    let (a, d) = expr_to_qrational(expr, var, pool)?;
    let g = poly_gcd(&a, &d);
    let (a, d) = if degree(&g) > 0 {
        (poly_divrem(&a, &g).0, poly_divrem(&d, &g).0)
    } else {
        (a, d)
    };
    let (a, d) = (trim(a), trim(d));
    if degree(&d) < 4 || a.is_empty() {
        return None;
    }
    let lambda = d.last()?.clone();
    let q: QPoly = d.iter().map(|c| Rational::from(c / &lambda)).collect();
    let p = top_square_root(&q)?;
    let k = trim(poly_sub(&q, &poly_mul(&p, &p)))
        .first()
        .cloned()
        .unwrap_or_default();
    if k <= 0 {
        return None;
    }
    // A = c·P′ (with D = λ(P² + k), so the integrand is (c/λ)·P′/(P² + k)).
    let dp = trim(poly_deriv(&p));
    if degree(&a) != degree(&dp) {
        return None;
    }
    let c = Rational::from(a.last()? / dp.last()?);
    let scaled: QPoly = dp.iter().map(|v| Rational::from(v * &c)).collect();
    if !trim(poly_sub(&a, &scaled)).is_empty() {
        return None;
    }
    let coeff = Rational::from(&c / &lambda);
    let k_expr = pool.rational(k.numer().clone(), k.denom().clone());
    let root_k = pool.func("sqrt", vec![k_expr]);
    let inv_root_k = pool.pow(root_k, pool.integer(-1_i32));
    let p_expr = qpoly_to_expr(&p, var, pool);
    let f = pool.mul(vec![
        pool.rational(coeff.numer().clone(), coeff.denom().clone()),
        inv_root_k,
        pool.func("atan", vec![pool.mul(vec![p_expr, inv_root_k])]),
    ]);
    Some(simplify(f, pool).value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::engine::verify_antiderivative_status;
    use crate::kernel::Domain;
    use std::collections::HashMap;

    fn check(src: &str) {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let mut syms = HashMap::from([("t".to_string(), t)]);
        let e = crate::parse::parse(src, &pool, &mut syms).unwrap();
        let f = atan_of_polynomial(e, t, &pool).unwrap_or_else(|| panic!("{src}"));
        assert!(
            verify_antiderivative_status(f, e, t, &pool).is_some(),
            "{src}: {}",
            pool.display(f)
        );
    }

    #[test]
    fn jeffrey_half_angle_integrands() {
        check("(2*t + 1)/(t^4 + 2*t^3 + t^2 + 1)");
        check("2*(3*t^2 - 2)/(t^6 - 4*t^4 + 4*t^2 + 1)");
        check("2*(6*t^2 - 1)/(4*t^6 - 4*t^4 + t^2 + 1)");
        check("(3*t^2 + 1)/((t^3 + t)^2 + 5)");
    }

    #[test]
    fn other_shapes_decline() {
        let pool = ExprPool::new();
        let t = pool.symbol("t", Domain::Real);
        let mut syms = HashMap::from([("t".to_string(), t)]);
        for src in [
            "t/(t^4 + 2*t^3 + t^2 + 1)",
            "(2*t + 1)/(t^4 + 2*t^3 + t^2 - 1)",
            "1/(t^2 + 1)",
        ] {
            let e = crate::parse::parse(src, &pool, &mut syms).unwrap();
            assert!(atan_of_polynomial(e, t, &pool).is_none(), "{src}");
        }
    }
}
