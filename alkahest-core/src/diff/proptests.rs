use super::{diff, diff_forward};
use crate::kernel::{Domain, ExprId, ExprPool};
use crate::poly::UniPoly;
use crate::simplify::engine::simplify;
use proptest::prelude::*;

fn small_coeff() -> impl Strategy<Value = i64> {
    -3i64..=3i64
}

/// Build a polynomial expression in `x` from a coefficient slice.
fn poly_expr(pool: &ExprPool, x: ExprId, coeffs: &[i64]) -> ExprId {
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
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// d/dx c == 0 for any constant expression (no x terms).
    #[test]
    fn diff_constant_is_zero(c in -100i64..=100i64) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.integer(c);
        let r = diff(expr, x, &pool).unwrap();
        prop_assert_eq!(r.value, pool.integer(0_i32));
    }

    /// Linearity: diff(a*f + b*g, x) == simplify(a*diff(f,x) + b*diff(g,x))
    #[test]
    fn diff_linearity(
        fa in proptest::collection::vec(small_coeff(), 1..=3),
        fb in proptest::collection::vec(small_coeff(), 1..=3),
        a in small_coeff(),
        b in small_coeff(),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let f = poly_expr(&pool, x, &fa);
        let g = poly_expr(&pool, x, &fb);
        let a_id = pool.integer(a);
        let b_id = pool.integer(b);

        // d/dx (a*f + b*g)
        let lhs_expr = pool.add(vec![pool.mul(vec![a_id, f]), pool.mul(vec![b_id, g])]);
        let lhs = diff(lhs_expr, x, &pool).unwrap();

        // a * d/dx f + b * d/dx g
        let df = diff(f, x, &pool).unwrap();
        let dg = diff(g, x, &pool).unwrap();
        let rhs_expr = pool.add(vec![
            pool.mul(vec![a_id, df.value]),
            pool.mul(vec![b_id, dg.value]),
        ]);
        let rhs = simplify(rhs_expr, &pool);

        // Compare as polynomials (handles representation differences)
        let lhs_poly = UniPoly::from_symbolic(lhs.value, x, &pool);
        let rhs_poly = UniPoly::from_symbolic(rhs.value, x, &pool);

        if let (Ok(lp), Ok(rp)) = (lhs_poly, rhs_poly) {
            prop_assert_eq!(
                lp.coefficients_i64(), rp.coefficients_i64(),
                "linearity failed for fa={:?}, fb={:?}, a={}, b={}", fa, fb, a, b
            );
        }
    }

    /// Product rule: diff(f*g) == f*diff(g) + g*diff(f) (simplified)
    #[test]
    fn diff_product_rule_check(
        fa in proptest::collection::vec(small_coeff(), 1..=3),
        fb in proptest::collection::vec(small_coeff(), 1..=3),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let f = poly_expr(&pool, x, &fa);
        let g = poly_expr(&pool, x, &fb);

        // d/dx (f*g)
        let fg = pool.mul(vec![f, g]);
        let lhs = diff(fg, x, &pool).unwrap();

        // f * g' + g * f'
        let df = diff(f, x, &pool).unwrap();
        let dg = diff(g, x, &pool).unwrap();
        let rhs_expr = pool.add(vec![
            pool.mul(vec![f, dg.value]),
            pool.mul(vec![g, df.value]),
        ]);
        let rhs = simplify(rhs_expr, &pool);

        let lhs_poly = UniPoly::from_symbolic(lhs.value, x, &pool);
        let rhs_poly = UniPoly::from_symbolic(rhs.value, x, &pool);

        if let (Ok(lp), Ok(rp)) = (lhs_poly, rhs_poly) {
            prop_assert_eq!(
                lp.coefficients_i64(), rp.coefficients_i64(),
                "product rule failed for fa={:?}, fb={:?}", fa, fb
            );
        }
    }

    /// Forward-mode and symbolic diff agree on polynomials.
    #[test]
    fn diff_forward_agrees_with_symbolic(
        coeffs in proptest::collection::vec(small_coeff(), 1..=5),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let expr = poly_expr(&pool, x, &coeffs);

        let sym = diff(expr, x, &pool).unwrap();
        let fwd = diff_forward(expr, x, &pool).unwrap();

        let sym_poly = UniPoly::from_symbolic(sym.value, x, &pool);
        let fwd_poly = UniPoly::from_symbolic(fwd.value, x, &pool);

        if let (Ok(sp), Ok(fp)) = (sym_poly, fwd_poly) {
            prop_assert_eq!(
                sp.coefficients_i64(), fp.coefficients_i64(),
                "forward vs symbolic mismatch for coeffs={:?}", coeffs
            );
        }
    }

    /// On random *shared* polynomial DAGs, `diff` equals the dense derivative
    /// — whichever route it takes: the root fast path (when the expansion is
    /// small), the rules (when it is not), and the rules forced by adding a
    /// foreign symbol `y` (so the input is no longer univariate).
    #[test]
    fn diff_of_shared_polynomial_dag_matches_dense_derivative(
        ops in proptest::collection::vec((0u8..3, 0usize..32, 0usize..32), 1..10),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut nodes = vec![x, pool.integer(2_i32), pool.integer(-3_i32), pool.integer(1_i32)];
        for (op, i, j) in ops {
            let a = nodes[i % nodes.len()];
            let b = nodes[j % nodes.len()];
            nodes.push(match op {
                0 => pool.add(vec![a, b]),
                1 => pool.mul(vec![a, b]),
                _ => pool.pow(a, pool.integer((j % 3) as i64 + 2)),
            });
        }
        let e = *nodes.last().unwrap();
        let dense = UniPoly::from_symbolic(e, x, &pool).unwrap();
        // Keep the reference computation cheap.
        prop_assume!(dense.degree() <= 200);
        let want = dense.derivative();

        let d = diff(e, x, &pool).unwrap();
        let got = UniPoly::from_symbolic(d.value, x, &pool).unwrap();
        prop_assert_eq!(&got, &want);

        let dy = diff(pool.add(vec![e, y]), x, &pool).unwrap();
        let got_y = UniPoly::from_symbolic(simplify(dy.value, &pool).value, x, &pool).unwrap();
        prop_assert_eq!(&got_y, &want);
    }
}
