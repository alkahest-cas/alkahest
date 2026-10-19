//! Lowering of the non-smooth functions to [`ExprData::Piecewise`].
//!
//! [`rewrite_piecewise`] replaces, bottom-up and everywhere in an expression,
//!
//! | node            | lowered form                                           |
//! |-----------------|--------------------------------------------------------|
//! | `abs(g)`        | `Piecewise((g ≥ 0, g); −g)`                            |
//! | `sign(g)`       | `Piecewise((g > 0, 1), (g < 0, −1); 0)`                |
//! | `heaviside(g)`  | `Piecewise((g > 0, 1), (g < 0, 0); 1/2)`               |
//! | `min(a, b)`     | `Piecewise((a ≤ b, a); b)`                             |
//! | `max(a, b)`     | `Piecewise((a ≥ b, a); b)`                             |
//!
//! `min`/`max` of more than two arguments are folded to the right.  The lowered
//! form takes the same value as the original at every point (the registry's
//! `sign(0) = 0` and `heaviside(0) = 1/2` included).  `floor`, `ceil` and
//! `round` have infinitely many pieces and are left alone.

use crate::kernel::{ExprData, ExprId, ExprPool, IdMap};

/// Lower `abs`, `sign`, `heaviside`, `min` and `max` in `expr` to `Piecewise`.
///
/// See the module documentation.  Expressions with none of those functions are
/// returned unchanged (same [`ExprId`]).
pub fn rewrite_piecewise(expr: ExprId, pool: &ExprPool) -> ExprId {
    let mut memo: IdMap<ExprId> = IdMap::default();
    lower(expr, pool, &mut memo)
}

fn lower(e: ExprId, pool: &ExprPool, memo: &mut IdMap<ExprId>) -> ExprId {
    if let Some(&r) = memo.get(&e) {
        return r;
    }
    let data = pool.get(e);
    let r = match data {
        ExprData::Add(a) => {
            let v: Vec<ExprId> = a.iter().map(|&x| lower(x, pool, memo)).collect();
            if v == a {
                e
            } else {
                pool.add(v)
            }
        }
        ExprData::Mul(a) => {
            let v: Vec<ExprId> = a.iter().map(|&x| lower(x, pool, memo)).collect();
            if v == a {
                e
            } else {
                pool.mul(v)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, x) = (lower(base, pool, memo), lower(exp, pool, memo));
            if (b, x) == (base, exp) {
                e
            } else {
                pool.pow(b, x)
            }
        }
        ExprData::Piecewise { branches, default } => {
            let nb: Vec<(ExprId, ExprId)> = branches
                .iter()
                .map(|&(c, v)| (lower(c, pool, memo), lower(v, pool, memo)))
                .collect();
            let nd = lower(default, pool, memo);
            if nb == branches && nd == default {
                e
            } else {
                pool.piecewise(nb, nd)
            }
        }
        ExprData::Predicate { kind, args } => {
            let v: Vec<ExprId> = args.iter().map(|&x| lower(x, pool, memo)).collect();
            if v == args {
                e
            } else {
                pool.predicate(kind, v)
            }
        }
        ExprData::Func { name, args } => {
            let v: Vec<ExprId> = args.iter().map(|&x| lower(x, pool, memo)).collect();
            lower_func(e, &name, &args, v, pool)
        }
        _ => e,
    };
    memo.insert(e, r);
    r
}

fn lower_func(e: ExprId, name: &str, old: &[ExprId], v: Vec<ExprId>, pool: &ExprPool) -> ExprId {
    let zero = pool.integer(0_i32);
    let one = pool.integer(1_i32);
    match (name, v.len()) {
        ("abs", 1) => {
            let g = v[0];
            let neg = pool.mul(vec![pool.integer(-1_i32), g]);
            pool.piecewise(vec![(pool.pred_ge(g, zero), g)], neg)
        }
        ("sign", 1) => {
            let g = v[0];
            pool.piecewise(
                vec![
                    (pool.pred_gt(g, zero), one),
                    (pool.pred_lt(g, zero), pool.integer(-1_i32)),
                ],
                zero,
            )
        }
        ("heaviside", 1) => {
            let g = v[0];
            pool.piecewise(
                vec![(pool.pred_gt(g, zero), one), (pool.pred_lt(g, zero), zero)],
                pool.rational(1, 2),
            )
        }
        ("min", n) | ("max", n) if n >= 2 => {
            let is_min = name == "min";
            let mut acc = v[n - 1];
            for k in (0..n - 1).rev() {
                let a = v[k];
                let cond = if is_min {
                    pool.pred_le(a, acc)
                } else {
                    pool.pred_ge(a, acc)
                };
                acc = pool.piecewise(vec![(cond, a)], acc);
            }
            acc
        }
        _ => {
            if v == old {
                e
            } else {
                pool.func(name.to_string(), v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    #[test]
    fn abs_min_max_lower_to_piecewise() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        for e in [
            pool.func("abs", vec![x]),
            pool.func("min", vec![x, pool.pow(x, pool.integer(2_i32))]),
            pool.func("max", vec![x, pool.integer(1_i32), pool.integer(3_i32)]),
            pool.func("sign", vec![x]),
        ] {
            let l = rewrite_piecewise(e, &pool);
            assert!(matches!(pool.get(l), ExprData::Piecewise { .. }));
        }
        assert_eq!(rewrite_piecewise(x, &pool), x);
    }
}
