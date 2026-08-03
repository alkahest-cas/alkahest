//! Picks a parallel simplification strategy from the shape of the expression.
//!
//! Feature-gated behind `--features parallel`.
//!
//! [`super::parallel::simplify_par`] and [`super::redex::simplify_redex`] win on
//! different shapes, and the difference is large — up to 4x either way — so the
//! choice matters more than tuning either one.  Fork-join keeps each chain on a
//! single worker and wins when the expression is wide, because a whole subtree
//! stays in one core's cache.  Level scheduling wins when it is deep, because
//! fork-join only forks on `Add`/`Mul` nodes with four or more children and a
//! deep chain gives it nothing to fork on.
//!
//! The advantage is also conditional on having cores to spend: at four workers
//! level scheduling was faster on every shape measured, including the widest,
//! so fork-join is only considered from [`MIN_WORKERS_FOR_FORK_JOIN`] up.
//!
//! The shape discriminator is average level width — nodes divided by height:
//!
//! | shape | nodes / height | pick |
//! |---|---|---|
//! | deep chain (2000 levels, width 1) | ~1 | level-scheduled |
//! | wide sum, independent terms (1024 x 8) | ~980 | fork-join |
//! | many medium chains (1024 x 32) | ~1000 | fork-join |
//!
//! # Caveat on the threshold
//!
//! [`WIDTH_THRESHOLD`] is calibrated on synthetic shapes, not on a profile of
//! real workloads.  The extremes it separates are unambiguous — a chain and a
//! wide sum differ by three orders of magnitude in average width — but where
//! exactly to cut between them is a guess, and expressions near the boundary
//! are close enough in cost that the choice matters little either way.  Treat
//! the constant as provisional until it can be checked against real traces.

#![cfg(feature = "parallel")]

use crate::deriv::log::DerivedExpr;
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::SimplifyConfig;

/// Average level width at or above which fork-join is preferred.
///
/// Fork-join only parallelises `Add`/`Mul` nodes with at least four children,
/// so it needs real width to beat level scheduling.
pub const WIDTH_THRESHOLD: f64 = 8.0;

/// Worker count below which level scheduling is preferred regardless of shape.
///
/// Fork-join's advantage on wide expressions comes from keeping a whole subtree
/// in one core's cache, and it only pays once there are enough cores to cover
/// the width. Measured on a 32-core machine, level scheduling was faster on
/// *every* shape at four workers — including the widest — while fork-join won
/// the wide shapes at sixteen and above.
pub const MIN_WORKERS_FOR_FORK_JOIN: usize = 8;

/// Which parallel simplifier [`simplify_auto`] selected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strategy {
    /// [`super::parallel::simplify_par`] — recursive fork-join.
    ForkJoin,
    /// [`super::redex::simplify_redex`] — level-scheduled redex bag.
    LevelScheduled,
}

/// Simplify with whichever parallel strategy suits the expression's shape.
///
/// Results are identical to [`crate::simplify::simplify`] either way; only the
/// schedule differs.  Note that the derivation log is deterministic only when
/// [`Strategy::LevelScheduled`] is chosen.
pub fn simplify_auto(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    simplify_auto_with_config(expr, pool, &SimplifyConfig::default())
}

/// Like [`simplify_auto`] but with a custom [`SimplifyConfig`].
pub fn simplify_auto_with_config(
    expr: ExprId,
    pool: &ExprPool,
    config: &SimplifyConfig,
) -> DerivedExpr<ExprId> {
    match choose_strategy(expr, pool) {
        Strategy::ForkJoin => super::parallel::simplify_par_with_config(expr, pool, config),
        Strategy::LevelScheduled => super::redex::simplify_redex_with_config(expr, pool, config),
    }
}

/// The strategy [`simplify_auto`] would use for `expr`.
///
/// Exposed so callers can log or override the decision, and so tests can check
/// it without timing anything.
pub fn choose_strategy(expr: ExprId, pool: &ExprPool) -> Strategy {
    // Below this many workers level scheduling wins whatever the shape, so
    // skip the shape probe entirely rather than pay a traversal to learn
    // something that cannot change the answer.
    if rayon::current_num_threads() < MIN_WORKERS_FOR_FORK_JOIN {
        return Strategy::LevelScheduled;
    }
    let (nodes, height) = shape(expr, pool);
    let average_width = nodes as f64 / height.max(1) as f64;
    if average_width >= WIDTH_THRESHOLD {
        Strategy::ForkJoin
    } else {
        Strategy::LevelScheduled
    }
}

/// Count the distinct nodes reachable from `root` and the height of the DAG.
///
/// Iterative, so it cannot overflow the stack on the deep expressions this is
/// meant to detect.  Costs one traversal, which is small next to the many
/// rule-matching passes that follow.
fn shape(root: ExprId, pool: &ExprPool) -> (usize, u32) {
    let n = pool.len();
    let mut height = vec![u32::MAX; n];
    let mut pushed = vec![false; n];
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];
    let mut nodes = 0_usize;
    let mut max_height = 0_u32;

    while let Some((id, expanded)) = stack.pop() {
        let i = id.0 as usize;
        if expanded {
            let h = pool.with(id, |data| {
                let mut h = 0_u32;
                for_each_child(data, |c| {
                    let ch = height[c.0 as usize];
                    debug_assert_ne!(ch, u32::MAX, "child measured after its parent");
                    h = h.max(ch.saturating_add(1));
                });
                h
            });
            height[i] = h;
            max_height = max_height.max(h);
            nodes += 1;
            continue;
        }
        if pushed[i] {
            continue;
        }
        pushed[i] = true;
        stack.push((id, true));
        pool.with(id, |data| {
            for_each_child(data, |c| {
                if !pushed[c.0 as usize] {
                    stack.push((c, false));
                }
            })
        });
    }

    // Height counts edges; a single node is one level deep.
    (nodes, max_height + 1)
}

/// Children a rewrite pass descends into, matching `simplify_children`.
fn for_each_child(data: &ExprData, mut f: impl FnMut(ExprId)) {
    match data {
        ExprData::Add(args) | ExprData::Mul(args) => args.iter().copied().for_each(f),
        ExprData::Func { args, .. } | ExprData::Predicate { args, .. } => {
            args.iter().copied().for_each(f)
        }
        ExprData::Pow { base, exp } => {
            f(*base);
            f(*exp);
        }
        ExprData::Piecewise { branches, default } => {
            branches.iter().for_each(|&(_, v)| f(v));
            f(*default);
        }
        ExprData::Forall { body, .. } | ExprData::Exists { body, .. } => f(*body),
        ExprData::BigO(arg) => f(*arg),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::simplify::simplify;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    fn junk(pool: &ExprPool, x: ExprId, depth: usize) -> ExprId {
        let one = pool.integer(1_i32);
        let zero = pool.integer(0_i32);
        let mut e = x;
        for _ in 0..depth {
            e = pool.mul(vec![e, one]);
            e = pool.add(vec![e, zero]);
        }
        e
    }

    /// Run inside a pool with enough workers that `choose_strategy` is deciding
    /// on shape rather than falling back to the low-worker rule.
    fn with_workers<R: Send>(f: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(MIN_WORKERS_FOR_FORK_JOIN)
            .build()
            .unwrap()
            .install(f)
    }

    #[test]
    fn deep_chain_picks_level_scheduling() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let deep = junk(&pool, x, 300);
        assert_eq!(
            with_workers(|| choose_strategy(deep, &pool)),
            Strategy::LevelScheduled
        );
    }

    #[test]
    fn wide_sum_picks_fork_join() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let args: Vec<ExprId> = (0..256)
            .map(|i| {
                let x = pool.symbol(format!("x{i}"), Domain::Real);
                pool.add(vec![x, zero])
            })
            .collect();
        let wide = pool.add(args);
        assert_eq!(
            with_workers(|| choose_strategy(wide, &pool)),
            Strategy::ForkJoin
        );
    }

    #[test]
    fn few_workers_pick_level_scheduling() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let args: Vec<ExprId> = (0..256)
            .map(|i| {
                let x = pool.symbol(format!("y{i}"), Domain::Real);
                pool.add(vec![x, zero])
            })
            .collect();
        let wide = pool.add(args);
        // Wide enough for fork-join on shape alone, but not enough workers.
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(MIN_WORKERS_FOR_FORK_JOIN - 1)
            .build()
            .unwrap();
        assert_eq!(
            tp.install(|| choose_strategy(wide, &pool)),
            Strategy::LevelScheduled
        );
    }

    /// Whichever branch is taken, the answer must match the sequential engine.
    #[test]
    fn auto_matches_sequential_on_both_shapes() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let deep = junk(&pool, x, 200);
        let zero = pool.integer(0_i32);
        let args: Vec<ExprId> = (0..64)
            .map(|i| {
                let s = pool.symbol(format!("z{i}"), Domain::Real);
                junk(&pool, s, 4)
            })
            .collect();
        let wide = pool.add(args);
        for expr in [deep, wide] {
            let seq = simplify(expr, &pool).value;
            let auto = with_workers(|| simplify_auto(expr, &pool).value);
            assert_eq!(seq, auto);
        }
        let _ = zero;
    }

    #[test]
    fn shape_measures_width_and_height() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // A chain of `Add(e, 0)` nodes: one node per level plus the leaves.
        let deep = junk(&pool, x, 10);
        let (nodes, height) = shape(deep, &pool);
        assert!(
            height >= 20,
            "chain should be at least 20 levels, got {height}"
        );
        assert!(
            (nodes as f64 / height as f64) < WIDTH_THRESHOLD,
            "a chain must read as narrow"
        );
    }
}
