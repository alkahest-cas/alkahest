//! Property tests for constant predicate folding and piecewise evaluation.

#[cfg(test)]
mod tests {
    use crate::jit::{compile, eval_interp};
    use crate::kernel::eval_const::try_predicate_bool_from_expr;
    use crate::kernel::expr::PredicateKind;
    use crate::kernel::subs::fold_predicates;
    use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn pool_xy() -> (ExprPool, ExprId, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        let y = p.symbol("y", Domain::Real);
        (p, x, y)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn try_predicate_bool_constant_ordering(a in -50i64..=50i64, b in -50i64..=50i64) {
            let p = ExprPool::new();
            let pred = p.pred_gt(p.integer(a), p.integer(b));
            let got = try_predicate_bool_from_expr(pred, &p);
            prop_assert_eq!(got, Some(a > b));
        }

        #[test]
        fn fold_predicates_constant_gt_is_true(a in 1i64..=100i64) {
            let p = ExprPool::new();
            let pred = p.pred_gt(p.integer(a), p.integer(0_i32));
            let folded = fold_predicates(pred, &p);
            let is_true = matches!(p.get(folded), ExprData::Predicate { kind: PredicateKind::True, .. });
            prop_assert!(is_true);
        }

        #[test]
        fn piecewise_interp_agrees_with_compiled(xv in -5.0f64..=5.0) {
            let (p, x, _) = pool_xy();
            let pw = p.piecewise(
                vec![(p.pred_gt(x, p.integer(0_i32)), x)],
                p.integer(-1_i32),
            );
            let f = compile(pw, &[x], &p).expect("compile piecewise");
            let mut env = HashMap::new();
            env.insert(x, xv);
            let direct = eval_interp(pw, &env, &p).expect("eval_interp piecewise");
            let compiled = f.call(&[xv]);
            prop_assert!((direct - compiled).abs() < 1e-10);
            prop_assert!(!compiled.is_nan());
        }
    }
}
