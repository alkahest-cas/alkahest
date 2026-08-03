//! Predicates on expression trees for noncommutative algebra (V3-2).

use crate::kernel::expr::ExprData;
use crate::kernel::pool::ExprPool;
use crate::kernel::ExprId;

/// `true` iff no non-commutative [`ExprData::Symbol`] appears anywhere in `expr`.
///
/// Used to decide whether multiplication may be canonically sorted or whether
/// rules like [`crate::simplify::rules::DivSelf`] may merge powers by base.
pub fn mult_tree_is_commutative(pool: &ExprPool, expr: ExprId) -> bool {
    // The flag is computed once, when `expr` is interned, from its children's
    // already-cached flags.  This used to walk the whole subtree on every call,
    // which made `ExprPool::mul` quadratic in the size of its argument: building
    // a nested product of depth 8000 took 373 ms against 2.5 ms for the
    // equivalent sum, and grew 4x for every doubling of depth.
    pool.is_mult_commutative(expr)
}

/// `true` iff some subtree is a symbol with `commutative == false`.
///
/// E-graph simplification assumes freely commuting numeric factors in its `Mul`
/// rules; we disable that backend when this predicate holds.
pub fn expr_contains_noncommutative_symbol(pool: &ExprPool, expr: ExprId) -> bool {
    pool.with(expr, |data| match data {
        ExprData::Symbol { commutative, .. } => !*commutative,
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => false,
        ExprData::Add(args) | ExprData::Mul(args) => args
            .iter()
            .any(|&c| expr_contains_noncommutative_symbol(pool, c)),
        ExprData::Pow { base, exp } => {
            expr_contains_noncommutative_symbol(pool, *base)
                || expr_contains_noncommutative_symbol(pool, *exp)
        }
        ExprData::Func { args, .. } => args
            .iter()
            .any(|&c| expr_contains_noncommutative_symbol(pool, c)),
        ExprData::Piecewise { branches, default } => {
            branches.iter().any(|(c, v)| {
                expr_contains_noncommutative_symbol(pool, *c)
                    || expr_contains_noncommutative_symbol(pool, *v)
            }) || expr_contains_noncommutative_symbol(pool, *default)
        }
        ExprData::Predicate { args, .. } => args
            .iter()
            .any(|&c| expr_contains_noncommutative_symbol(pool, c)),
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
            expr_contains_noncommutative_symbol(pool, *var)
                || expr_contains_noncommutative_symbol(pool, *body)
        }
        ExprData::BigO(inner) => expr_contains_noncommutative_symbol(pool, *inner),
        ExprData::RootSum { poly, body, .. } => {
            expr_contains_noncommutative_symbol(pool, *poly)
                || expr_contains_noncommutative_symbol(pool, *body)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// The pre-cache implementation, kept here as an oracle: walk the whole
    /// subtree every time.
    fn reference(pool: &ExprPool, expr: ExprId) -> bool {
        pool.with(expr, |data| match data {
            ExprData::Symbol { commutative, .. } => *commutative,
            ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
            ExprData::Add(args) | ExprData::Mul(args) => args.iter().all(|&c| reference(pool, c)),
            ExprData::Pow { base, exp } => reference(pool, *base) && reference(pool, *exp),
            ExprData::Func { args, .. } => args.iter().all(|&c| reference(pool, c)),
            ExprData::Piecewise { branches, default } => {
                branches
                    .iter()
                    .all(|(c, v)| reference(pool, *c) && reference(pool, *v))
                    && reference(pool, *default)
            }
            ExprData::Predicate { args, .. } => args.iter().all(|&c| reference(pool, c)),
            ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
                reference(pool, *var) && reference(pool, *body)
            }
            ExprData::BigO(inner) => reference(pool, *inner),
            ExprData::RootSum { poly, body, .. } => {
                reference(pool, *poly) && reference(pool, *body)
            }
        })
    }

    /// The cached flag must agree with a full subtree walk everywhere,
    /// including when a non-commutative generator is buried deep.
    #[test]
    fn cached_flag_matches_full_walk() {
        let pool = ExprPool::new();
        let c = pool.symbol("c", Domain::Real);
        let nc = pool.symbol_commutative("nc", Domain::Real, false);
        let two = pool.integer(2_i32);

        let mut nodes = vec![c, nc, two];
        // Commutative-only subtree.
        let pure = pool.add(vec![c, two]);
        nodes.push(pure);
        nodes.push(pool.pow(pure, two));
        nodes.push(pool.func("sin", vec![pure]));
        // Same shapes with a non-commutative generator buried inside.
        let tainted = pool.add(vec![nc, two]);
        nodes.push(tainted);
        nodes.push(pool.pow(tainted, two));
        nodes.push(pool.func("sin", vec![tainted]));
        nodes.push(pool.mul(vec![pure, tainted]));
        nodes.push(pool.big_o(tainted));
        nodes.push(pool.pred_lt(tainted, pure));
        // Deeply nested, so a stale flag would show up.
        let mut deep = c;
        for _ in 0..50 {
            deep = pool.mul(vec![deep, two]);
            nodes.push(deep);
        }
        let mut deep_nc = nc;
        for _ in 0..50 {
            deep_nc = pool.mul(vec![deep_nc, two]);
            nodes.push(deep_nc);
        }

        for id in nodes {
            assert_eq!(
                mult_tree_is_commutative(&pool, id),
                reference(&pool, id),
                "cached flag disagrees with full walk for {}",
                crate::kernel::display::render_unicode(id, &pool)
            );
        }
    }

    #[test]
    fn noncommutative_blocks_canonical_sorting() {
        let pool = ExprPool::new();
        let a = pool.symbol_commutative("a", Domain::Real, false);
        let b = pool.symbol_commutative("b", Domain::Real, false);
        // `mul` may not sort these, so the two orders stay distinct.
        assert_ne!(pool.mul(vec![a, b]), pool.mul(vec![b, a]));
        assert!(!mult_tree_is_commutative(&pool, pool.mul(vec![a, b])));
    }
}
