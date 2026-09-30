//! Predicates on expression trees for noncommutative algebra (V3-2).

use crate::kernel::expr::ExprData;
use crate::kernel::pool::{ExprPool, POS_INFINITY_SYMBOL};
use crate::kernel::ExprId;

/// `true` iff no non-commutative [`ExprData::Symbol`](crate::kernel::ExprData::Symbol)
/// appears anywhere in `expr`.
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
///
/// This is exactly the negation of [`mult_tree_is_commutative`]: both follow
/// the same recurrence (a symbol contributes its own flag, numbers commute,
/// every other node combines all of its children — `RootSum` skipping its
/// bound `var` in both), so the answer is the O(1) flag cached at intern time
/// rather than a walk that visited a shared subterm once per path to it.
pub fn expr_contains_noncommutative_symbol(pool: &ExprPool, expr: ExprId) -> bool {
    !pool.is_mult_commutative(expr)
}

/// `true` iff `expr` is itself a value that is not a finite number: the
/// canonical `∞` symbol of [`ExprPool::pos_infinity`], or a `Float` holding
/// `±inf` or `NaN`.
///
/// The algebraic identities the simplifier leans on — `x − x = 0`,
/// `0 · x = 0`, `x / x = 1` — hold for every finite `x` and fail for exactly
/// these: `∞ − ∞`, `0 · ∞` and `∞ / ∞` are indeterminate, and IEEE `NaN` is
/// not even equal to itself.  An ordinary symbol is *not* reported here: its
/// value is a finite number by the library's standing convention, and the
/// rules keep treating it that way.
///
/// O(1): one node probe.
pub fn is_non_finite_atom(pool: &ExprPool, expr: ExprId) -> bool {
    pool.with(expr, |d| match d {
        ExprData::Symbol { name, .. } => name == POS_INFINITY_SYMBOL,
        ExprData::Float(f) => !f.inner.is_finite(),
        _ => false,
    })
}

/// `true` iff some node of `expr` satisfies [`is_non_finite_atom`].
///
/// O(1): the flag is computed bottom-up when a node is interned, like
/// [`mult_tree_is_commutative`]'s.
pub fn contains_non_finite_atom(pool: &ExprPool, expr: ExprId) -> bool {
    pool.has_non_finite(expr)
}

/// The subtree walk `expr_contains_noncommutative_symbol` used to perform,
/// kept as its test oracle.
#[cfg(test)]
fn contains_noncommutative_walk(pool: &ExprPool, expr: ExprId) -> bool {
    let rec = |c: ExprId| contains_noncommutative_walk(pool, c);
    pool.with(expr, |data| match data {
        ExprData::Symbol { commutative, .. } => !*commutative,
        ExprData::Integer(_) | ExprData::Rational(_) | ExprData::Float(_) => false,
        ExprData::Add(args) | ExprData::Mul(args) => args.iter().any(|&c| rec(c)),
        ExprData::Pow { base, exp } => rec(*base) || rec(*exp),
        ExprData::Func { args, .. } => args.iter().any(|&c| rec(c)),
        ExprData::Piecewise { branches, default } => {
            branches.iter().any(|(c, v)| rec(*c) || rec(*v)) || rec(*default)
        }
        ExprData::Predicate { args, .. } => args.iter().any(|&c| rec(c)),
        ExprData::Forall { var, body } | ExprData::Exists { var, body } => rec(*var) || rec(*body),
        ExprData::BigO(inner) => rec(*inner),
        ExprData::RootSum { poly, body, .. } => rec(*poly) || rec(*body),
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

    /// `expr_contains_noncommutative_symbol` now reads the cached flag; it
    /// must agree with the subtree walk it replaced on every node kind.
    #[test]
    fn contains_noncommutative_matches_walk() {
        let pool = ExprPool::new();
        let c = pool.symbol("c", Domain::Real);
        let nc = pool.symbol_commutative("nc", Domain::Real, false);
        let two = pool.integer(2_i32);
        let half = pool.rational(1, 2);
        let f = pool.float(0.5, 53);
        let mut nodes = vec![c, nc, two, half, f];
        for leaf in [c, nc] {
            let s = pool.add(vec![leaf, two]);
            nodes.push(s);
            nodes.push(pool.mul(vec![s, half]));
            nodes.push(pool.pow(two, s));
            nodes.push(pool.func("atan2", vec![c, s]));
            nodes.push(pool.pred_lt(s, f));
            nodes.push(pool.piecewise(vec![(pool.pred_gt(c, two), s)], two));
            nodes.push(pool.piecewise(vec![(pool.pred_gt(s, two), c)], two));
            nodes.push(pool.piecewise(vec![(pool.pred_gt(c, two), c)], s));
            nodes.push(pool.forall(c, pool.pred_ge(s, two)));
            nodes.push(pool.exists(c, pool.pred_ge(s, two)));
            nodes.push(pool.big_o(s));
            nodes.push(pool.root_sum(s, c, c));
            nodes.push(pool.root_sum(c, c, s));
            // A non-commutative *bound* variable is skipped by both.
            nodes.push(pool.root_sum(c, leaf, c));
        }
        for id in nodes {
            assert_eq!(
                expr_contains_noncommutative_symbol(&pool, id),
                contains_noncommutative_walk(&pool, id),
                "cached flag disagrees with walk for {}",
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
