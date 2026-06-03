use super::engine::simplify;
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;
use proptest::prelude::*;

fn small_coeff() -> impl Strategy<Value = i64> {
    -5i64..=5i64
}

/// Build a random polynomial expression tree in `x` from coefficient slices.
/// Returns the ExprId of the expression in the pool.
fn poly_expr(pool: &ExprPool, x: ExprId, coeffs: &[i64]) -> ExprId {
    // p(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
    let mut terms: Vec<ExprId> = vec![];
    for (i, &c) in coeffs.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let c_id = pool.integer(c);
        if i == 0 {
            terms.push(c_id);
        } else {
            let deg = pool.integer(i as i32);
            let xpow = pool.pow(x, deg);
            if c == 1 {
                terms.push(xpow);
            } else {
                terms.push(pool.mul(vec![c_id, xpow]));
            }
        }
    }
    match terms.len() {
        0 => pool.integer(0_i32),
        1 => terms[0],
        _ => pool.add(terms),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn simplify_idempotent(
        coeffs in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let first = simplify(expr, &pool);
        let second = simplify(first.value, &pool);
        prop_assert_eq!(
            first.value, second.value,
            "simplify(simplify(e)) != simplify(e) for coeffs={:?}", coeffs
        );
    }

    #[test]
    fn simplify_constant_zero_is_zero(
        coeffs in proptest::collection::vec(Just(0i64), 1..=4),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let r = simplify(expr, &pool);
        // All-zero polynomial should simplify to Integer(0)
        prop_assert_eq!(r.value, pool.integer(0_i32));
    }

    #[test]
    fn simplify_integer_constant_folds(a in -100i64..=100i64, b in -100i64..=100i64) {
        let pool = ExprPool::new();
        let expr = pool.add(vec![pool.integer(a), pool.integer(b)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(a + b));
    }

    #[test]
    fn simplify_mul_constant_folds(a in -20i64..=20i64, b in -20i64..=20i64) {
        let pool = ExprPool::new();
        let expr = pool.mul(vec![pool.integer(a), pool.integer(b)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(a * b));
    }

    /// sqrt(n) → integer root when n is a perfect square (n > 0).
    #[test]
    fn simplify_sqrt_perfect_square(n in 1u32..=50u32) {
        let pool = ExprPool::new();
        let n_sq = (n as i64) * (n as i64);
        let expr = pool.func("sqrt", vec![pool.integer(n_sq)]);
        let r = simplify(expr, &pool);
        prop_assert_eq!(r.value, pool.integer(n as i64));
    }

    #[test]
    fn simplify_preserves_polynomial_value(
        coeffs in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        // Convert to UniPoly before and after simplification; coefficients must match.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);
        let r = simplify(expr, &pool);

        // Both should convert to polynomial in x (if they're polynomials)
        let poly_before = UniPoly::from_symbolic(expr, x, &pool);
        let poly_after = UniPoly::from_symbolic(r.value, x, &pool);
        if let (Ok(pb), Ok(pa)) = (poly_before, poly_after) {
            prop_assert_eq!(
                pb.coefficients_i64(), pa.coefficients_i64(),
                "polynomial changed under simplification"
            );
        }
    }
}
