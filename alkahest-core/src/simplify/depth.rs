//! The depth ceiling as the simplification entry points actually need it.
//!
//! [`crate::kernel::MAX_EXPR_DEPTH`] is one number for every walker in the
//! crate, calibrated against the shallowest one that recurses without a net.
//! The simplification traversals are not that.  `simplify::engine`'s
//! `simplify_node` and `simplify_node_indexed` run under the segmented-stack
//! trampoline in `simplify::stack`, which continues the recursion on a freshly
//! spawned, larger-stacked thread before the current one is spent.  For those
//! the depth bound is removed rather than lowered, so applying a 2 048-level
//! ceiling to them costs capability and buys nothing.
//!
//! Neither of the two `parallel`-gated strategies changes that, and both had
//! to be checked rather than assumed: `simplify::parallel`'s
//! `simplify_node_par` goes through the same trampoline, and
//! `simplify::redex` buckets the DAG by height and rewrites a level at a time,
//! so it does not recurse at all.  With the feature off, both entry points
//! fall back to `simplify::engine`.  Every branch of every `cfg` on this path
//! is therefore stack-safe, which is what makes it safe to lift the ceiling
//! from an entry point whose implementation depends on a feature flag.
//!
//! One thing on that path is still a plain recursion, though, and this module
//! is what keeps it from being reached.
//!
//! # The one route that is still bounded by the stack
//!
//! Every default simplification pass ends the same way.  `simplify_with`,
//! `simplify_par_with_config` and `simplify_redex_with_config` each collect
//! the facts implied by static symbol domains — a [`Domain::Positive`] or
//! [`Domain::NonZero`] symbol anywhere in the result — and, if there are any,
//! hand the expression to [`crate::simplify::colored_egraph`] for an
//! assumption-gated pass.  That pass is *not* trampolined:
//! `ColoredEgraph::from_expr`'s node collection and `ColoredEgraph::rebuild`
//! both descend one native frame per level, and on an 8 MiB stack a release
//! build segfaults somewhere between 60 000 and 100 000 levels.  It is also
//! quadratic in node count — a 5 000-level chain took 4.8 s, a 20 000-level
//! one 100 s — so "make it trampolined" would trade a crash for a hang.
//!
//! So the ceiling still has to apply, but only to the expressions that reach
//! that pass.  [`check_simplify_depth`] is that test: past
//! [`MAX_EXPR_DEPTH`](crate::kernel::MAX_EXPR_DEPTH) it asks whether this
//! expression would take the colored route, and refuses only if it would.
//!
//! # Why the predicate is the real collector rather than a cheaper copy
//!
//! [`check_simplify_depth`] decides by running
//! `simplify::assumptions`'s own `collect_static_domain_facts` and looking at
//! whether it produced anything.  A hand-written "does this contain a
//! positive symbol" scan would be faster and would drift: the moment a new
//! `Domain` or node kind starts contributing a fact, the guard and the thing
//! it guards would disagree, and the disagreement that matters — guard says
//! no, simplifier says yes — is a segfault.  Sharing the collector makes that
//! impossible by construction.
//!
//! The cost is one extra iterative walk of an expression already past 2 048
//! levels deep.  Expressions at or under the ceiling never reach it: the depth
//! comparison is checked first and is a single array read, so the hot path
//! pays exactly what it paid before.
//!
//! # What this does not cover
//!
//! * **The e-graph simplifiers.** `simplify_egraph` and `simplify_colored`
//!   are separate engines with their own unbounded recursion, reached through
//!   their own entry points.  They keep [`check_expr_depth`] and are not this
//!   function's business.
//! * **Rendering the derivation log.** The log a simplification returns holds
//!   `ExprId`s, and `alkahest-py` renders them with the same recursive printer
//!   `str()` uses, which overflows around 24 500 levels.  That is handled
//!   where it happens, in `alkahest-py`'s `make_derived_result`, by recording
//!   an over-deep step's depth instead of its text.
//! * **Explicit assumption contexts.** A caller who passes an
//!   [`AssumptionContext`](crate::simplify::AssumptionContext) has asked for
//!   the colored pass outright, so its entry point keeps the unconditional
//!   [`check_expr_depth`].
//!
//! [`Domain::Positive`]: crate::kernel::Domain::Positive
//! [`Domain::NonZero`]: crate::kernel::Domain::NonZero

use crate::deriv::SideCondition;
use crate::kernel::depth::{check_expr_depth, DepthLimitError};
use crate::kernel::{ExprId, ExprPool};

/// Refuse `id` only if simplifying it would recurse without a net.
///
/// The bottom-up simplification traversals are stack-safe at any depth (see
/// the [module documentation](self)), so this accepts expressions far past
/// [`MAX_EXPR_DEPTH`](crate::kernel::MAX_EXPR_DEPTH) — a 100 000-level `sin`
/// chain simplifies in about 0.1 s.  What it still refuses is a too-deep
/// expression that would be handed to the assumption-gated colored e-graph
/// pass, which is a plain recursion: that is an expression past the ceiling
/// containing a `Domain::Positive` or `Domain::NonZero` symbol.
///
/// The error, when there is one, is the same [`DepthLimitError`] with the same
/// `E-DEPTH-001` code and the same `limit` as [`check_expr_depth`]; a caller
/// cannot tell the two guards apart, and does not need to.
///
/// O(1) at or below the ceiling.  Past it, one iterative walk of the
/// expression — no recursion, so this cannot overflow the stack in the course
/// of deciding whether something else would.
///
/// ```
/// use alkahest_cas::kernel::{Domain, ExprPool, MAX_EXPR_DEPTH};
/// use alkahest_cas::simplify::check_simplify_depth;
///
/// let pool = ExprPool::new();
///
/// // Deep, but nothing in it routes to the colored pass: accepted.
/// let x = pool.symbol("x", Domain::Real);
/// let mut deep = x;
/// for _ in 0..MAX_EXPR_DEPTH + 100 {
///     deep = pool.func("sin", vec![deep]);
/// }
/// assert!(check_simplify_depth(&pool, deep).is_ok());
///
/// // The same shape over a positive symbol takes the recursive route.
/// let p = pool.symbol("p", Domain::Positive);
/// let mut deep_positive = p;
/// for _ in 0..MAX_EXPR_DEPTH + 100 {
///     deep_positive = pool.func("sin", vec![deep_positive]);
/// }
/// let err = check_simplify_depth(&pool, deep_positive).unwrap_err();
/// assert_eq!(err.limit, MAX_EXPR_DEPTH);
/// ```
pub fn check_simplify_depth(pool: &ExprPool, id: ExprId) -> Result<(), DepthLimitError> {
    // Cheap first: at or under the ceiling nothing can go wrong on any route,
    // and this is the hot path every ordinary `simplify` call takes.
    let shallow_enough = check_expr_depth(pool, id);
    if shallow_enough.is_ok() {
        return Ok(());
    }
    if takes_colored_route(id, pool) {
        return shallow_enough;
    }
    Ok(())
}

/// Whether a default simplification of `expr` would end in the colored
/// e-graph pass.
///
/// Deliberately the same collector the simplifier itself uses, so the two
/// cannot disagree about which expressions take that route.
fn takes_colored_route(expr: ExprId, pool: &ExprPool) -> bool {
    let mut facts: Vec<SideCondition> = Vec::new();
    super::assumptions::collect_static_domain_facts(expr, pool, &mut facts);
    !facts.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::AlkahestError;
    use crate::kernel::{Domain, MAX_EXPR_DEPTH};

    /// `depth` levels of `sin` over `leaf`.
    fn chain(pool: &ExprPool, leaf: ExprId, depth: u32) -> ExprId {
        let mut acc = leaf;
        for _ in 0..depth {
            acc = pool.func("sin", vec![acc]);
        }
        acc
    }

    /// The capability this whole module exists to hand back: past the ceiling
    /// and still accepted, because nothing on the route recurses.
    #[test]
    fn deep_is_accepted_when_no_assumption_pass_will_run() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let deep = chain(&pool, x, MAX_EXPR_DEPTH * 8);
        assert!(pool.depth(deep) > MAX_EXPR_DEPTH);
        assert!(check_simplify_depth(&pool, deep).is_ok());
        // …and the unconditional guard would have refused exactly this.
        assert!(check_expr_depth(&pool, deep).is_err());
    }

    /// A positive symbol anywhere in the tree routes the whole expression into
    /// the colored pass, which is a plain recursion — so the ceiling has to
    /// come back, whatever else the expression is made of.
    #[test]
    fn deep_is_refused_when_a_static_domain_fact_routes_it_to_the_egraph() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let p = pool.symbol("p", Domain::Positive);
        // The positive symbol is a leaf of one branch only; the guard must
        // still see it.
        let deep = chain(&pool, x, MAX_EXPR_DEPTH * 2);
        let mixed = pool.add(vec![deep, p]);
        let err = check_simplify_depth(&pool, mixed).expect_err("must be refused");
        assert_eq!(err.limit, MAX_EXPR_DEPTH);
        assert_eq!(err.code(), "E-DEPTH-001");
    }

    /// `Domain::NonZero` authorizes rewrites too, so it routes the same way.
    #[test]
    fn a_nonzero_symbol_routes_the_same_way_as_a_positive_one() {
        let pool = ExprPool::new();
        let nz = pool.symbol("nz", Domain::NonZero);
        let deep = chain(&pool, nz, MAX_EXPR_DEPTH * 2);
        assert!(check_simplify_depth(&pool, deep).is_err());
    }

    /// Below the ceiling the colored pass is reached at a depth its recursion
    /// survives, so a positive symbol must not make an ordinary expression
    /// unsimplifiable.
    #[test]
    fn a_shallow_expression_is_accepted_however_it_would_route() {
        let pool = ExprPool::new();
        let p = pool.symbol("p", Domain::Positive);
        let shallow = chain(&pool, p, MAX_EXPR_DEPTH - 1);
        assert_eq!(pool.depth(shallow), MAX_EXPR_DEPTH);
        assert!(check_simplify_depth(&pool, shallow).is_ok());
        assert!(check_expr_depth(&pool, shallow).is_ok());
    }

    /// The guard and the simplifier must agree about which expressions take
    /// the colored route: the guard refusing one the simplifier would not send
    /// there costs capability, and accepting one it would send there is a
    /// segfault.  Asserted against the collector's own output rather than a
    /// restatement of it.
    #[test]
    fn the_predicate_agrees_with_the_collector_the_simplifier_uses() {
        let pool = ExprPool::new();
        for domain in [
            Domain::Real,
            Domain::Complex,
            Domain::Integer,
            Domain::NonNegative,
            Domain::Positive,
            Domain::NonZero,
        ] {
            let leaf = pool.symbol(format!("s_{domain:?}"), domain);
            let deep = chain(&pool, leaf, MAX_EXPR_DEPTH + 1);
            let mut facts = Vec::new();
            crate::simplify::assumptions::collect_static_domain_facts(deep, &pool, &mut facts);
            assert_eq!(
                check_simplify_depth(&pool, deep).is_err(),
                !facts.is_empty(),
                "guard and collector disagree for {domain:?}"
            );
        }
    }
}
