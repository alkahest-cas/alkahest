//! Predicates on expression trees for noncommutative algebra (V3-2).

use crate::kernel::expr::ExprData;
use crate::kernel::pool::ExprPool;
use crate::kernel::ExprId;

/// `true` iff no non-commutative [`ExprData::Symbol`] appears anywhere in `expr`.
///
/// Used to decide whether multiplication may be canonically sorted or whether
/// rules like [`crate::simplify::rules::DivSelf`] may merge powers by base.
pub fn mult_tree_is_commutative(pool: &ExprPool, expr: ExprId) -> bool {
    pool.with(expr, |data| match data {
        ExprData::Symbol { commutative, .. } => *commutative,
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => true,
        ExprData::Add(args) | ExprData::Mul(args) => {
            args.iter().all(|&c| mult_tree_is_commutative(pool, c))
        }
        ExprData::Pow { base, exp } => {
            mult_tree_is_commutative(pool, *base) && mult_tree_is_commutative(pool, *exp)
        }
        ExprData::Func { args, .. } => args.iter().all(|&c| mult_tree_is_commutative(pool, c)),
        ExprData::Piecewise { branches, default } => {
            branches.iter().all(|(c, v)| {
                mult_tree_is_commutative(pool, *c) && mult_tree_is_commutative(pool, *v)
            }) && mult_tree_is_commutative(pool, *default)
        }
        ExprData::Predicate { args, .. } => args.iter().all(|&c| mult_tree_is_commutative(pool, c)),
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => {
            mult_tree_is_commutative(pool, *var) && mult_tree_is_commutative(pool, *body)
        }
        ExprData::BigO(inner) => mult_tree_is_commutative(pool, *inner),
    })
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
    })
}
