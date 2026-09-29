//! Level-scheduled parallel simplification — an HVM2-style redex bag.
//!
//! Feature-gated behind `--features parallel`.
//!
//! # Where this comes from
//!
//! HVM2 evaluates a program by reducing *interaction net* redexes.  Two ideas
//! from that design transfer to a term-rewriting CAS, even though interaction
//! nets themselves do not:
//!
//! 1. **A flat array with atomic links instead of a locked side table.**
//!    The memo is a `Vec<AtomicU32>` with one cell per node reachable from
//!    the root — no shard locks, one relaxed load per child.  Which cell
//!    belongs to which `ExprId` is fixed before any worker starts, so that map
//!    is read-only while the pass runs and costs one uncontended hash per
//!    child.  (Indexing the array by `ExprId` directly would drop even that
//!    hash, but sizes the array by the whole pool rather than the expression —
//!    see `Table`.)  [`super::parallel`] uses a `DashMap`, which costs two
//!    hashes per node visit and contends across workers.
//!
//! 2. **A bag of independent redexes instead of a recursive fork-join.**
//!    [`super::parallel`] mirrors the sequential traversal and only forks on
//!    `Add`/`Mul` nodes with four or more children; every other node is
//!    handled by the calling thread, so the available parallelism is a
//!    property of the expression's *shape*.  Here the DAG is bucketed by
//!    height first, and every node at a given height is an independent redex
//!    because all of its children live at lower heights.  Each level is one
//!    `par_iter`, so the parallelism is the *width* of the level regardless of
//!    node type.
//!
//! What does not transfer is HVM2's evaluation model: alkahest's hot paths are
//! bignum and finite-field arithmetic behind FLINT/Arb, which cannot run inside
//! an interaction net.  This module borrows the scheduler, not the runtime.
//!
//! # Consequences
//!
//! * The traversal is iterative, so deep expressions cannot overflow the stack.
//!   [`super::parallel`] needs a stack-refill trampoline; this does not.
//! * Each node is visited exactly once per pass, so derivation logs are
//!   deduplicated by construction, and merging them in level order makes the
//!   log **deterministic** — unlike the fork-join path, where two workers can
//!   race to simplify the same node and log it twice.
//! * Results are identical to [`crate::simplify::simplify`]; the rule list,
//!   the bottom-up order and the fixed-point loop are the same.
//!
//! # When to prefer which
//!
//! This is *not* a strict improvement on [`super::parallel`] — the two win on
//! different shapes.  Best time over 1–32 threads on a 32-core machine
//! (`examples/simplify_three_way.rs`, which times the call only):
//!
//! | shape | sequential | best `simplify_par` | best `simplify_redex` |
//! |---|---|---|---|
//! | deep chain (2000 levels, width 1) | 38.7 ms | 23.1 ms | **5.5 ms** |
//! | wide sum, independent terms (1024 × depth 8) | 28.4 ms | **5.1 ms** | 10.3 ms |
//! | many medium chains (1024 × depth 32) | 115.6 ms | **19.2 ms** | 36.6 ms |
//! | wide sum over a shared DAG (1024 terms) | 2.48 ms | 0.89 ms | **0.83 ms** |
//!
//! Fork-join keeps each chain on one worker, so wide sums of independent terms
//! get good cache locality; level scheduling sweeps every chain at every
//! height, streaming the whole DAG through cache once per level.  Deep
//! expressions invert that: fork-join finds no wide `Add`/`Mul` to fork on, so
//! it runs essentially sequentially *and* pays for the stack trampoline, while
//! level scheduling still runs flat.
//!
//! At one thread this path is faster than fork-join on every shape measured
//! (67.2 ms → 56.7 ms on a 4096-term sum; 121.5 ms → 90.3 ms on the chains;
//! 23.3 ms → 10.1 ms on the deep chain), because a flat array index is cheaper
//! than a `DashMap` probe and the traversal never recurses.
//!
//! A barrier-free variant — per-node counters of unreduced children, firing
//! each node the moment its last child lands, which is what HVM2 actually does
//! — was implemented and measured against this one under an identical harness.
//! It was never reliably faster on any workload, so it was deleted rather than
//! carried as dead weight: the per-level barrier is not the bottleneck,
//! per-node work and cache locality are.

#![cfg(feature = "parallel")]

use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::{rules_for_config, SimplifyConfig};
use crate::simplify::idmap::IdMap;
use crate::simplify::rules::RewriteRule;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Marks a slot in the substitution table that has not been written yet.
/// `ExprId` is a `u32` index, so this can never collide with a real id in a
/// pool that fits in memory.
const UNMAPPED: u32 = u32::MAX;

/// Levels narrower than this are simplified on the calling thread: below it,
/// rayon's fork/join costs more than the work it distributes.
const PAR_LEVEL_THRESHOLD: usize = 8;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Simplify `expr` by scheduling independent redexes level by level.
///
/// Equivalent to [`crate::simplify::simplify`], including the derivation log,
/// which is deterministic here regardless of thread count.
pub fn simplify_redex(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    simplify_redex_with_config(expr, pool, &SimplifyConfig::default())
}

/// Like [`simplify_redex`] but with a custom [`SimplifyConfig`].
pub fn simplify_redex_with_config(
    expr: ExprId,
    pool: &ExprPool,
    config: &SimplifyConfig,
) -> DerivedExpr<ExprId> {
    if config.expand {
        crate::simplify::rules::clear_expand_limits();
    }
    let rules = rules_for_config(config);
    let mut current = expr;
    let mut full_log = DerivationLog::new();

    for _ in 0..config.max_iterations {
        let (value, log) = one_pass(current, pool, &rules);
        full_log = full_log.merge(log);
        if value == current {
            break;
        }
        current = value;
    }

    if config.expand {
        // Same as `simplify_with`: a bound that stopped an expansion is part of
        // what happened, and belongs in the log rather than nowhere.
        full_log = full_log.merge(crate::simplify::engine::expand_limit_log());
    }

    // Assumption-driven (colored e-graph) pass, mirroring `simplify_with`.
    let mut assumptions = config.assumptions.clone();
    super::assumptions::collect_static_domain_facts(current, pool, &mut assumptions);
    if !assumptions.is_empty() {
        let colored = super::colored_egraph::apply_colored_if_needed(current, pool, &assumptions);
        return DerivedExpr::with_log(colored.value, full_log.merge(colored.log));
    }
    DerivedExpr::with_log(current, full_log)
}

// ---------------------------------------------------------------------------
// One bottom-up pass
// ---------------------------------------------------------------------------

fn one_pass(
    root: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
) -> (ExprId, DerivationLog) {
    // Substitution table: original id → simplified id, with one cell per node
    // reachable from `root`.  Rewrites intern ids outside that set, but those
    // are only ever written as values, never used as keys.
    let (levels, table) = build_levels(root, pool);

    let mut log = DerivationLog::new();
    for level in &levels {
        // Each node in a level is written by exactly one task, and every child
        // lives in a lower level that has already been joined, so relaxed
        // ordering suffices: the join between levels supplies the fence.
        let logs: Vec<DerivationLog> = if level.len() >= PAR_LEVEL_THRESHOLD {
            level
                .par_iter()
                .map(|&id| reduce_node(id, pool, rules, &table))
                .collect()
        } else {
            level
                .iter()
                .map(|&id| reduce_node(id, pool, rules, &table))
                .collect()
        };
        // Merged in level order, so the log does not depend on the schedule.
        for l in logs {
            log = log.merge(l);
        }
    }

    (lookup(&table, root), log)
}

/// Rewrite one node, given that all of its children are already mapped.
fn reduce_node(
    id: ExprId,
    pool: &ExprPool,
    rules: &[Box<dyn RewriteRule>],
    table: &Table,
) -> DerivationLog {
    let rebuilt = pool.with(id, |data| rebuild(id, data, pool, table));
    let (current, log) = crate::simplify::engine::apply_rules(rebuilt, pool, rules);
    table.cell(id).store(current.0, Ordering::Relaxed);
    log
}

fn lookup(table: &Table, id: ExprId) -> ExprId {
    let mapped = table.cell(id).load(Ordering::Relaxed);
    debug_assert_ne!(mapped, UNMAPPED, "child reduced out of level order");
    ExprId(mapped)
}

/// The substitution table for one pass: a dense array of atomic cells, one
/// per node reachable from the pass's root, and the map from each such node
/// to its cell.
///
/// This used to be a `Vec<AtomicU32>` indexed directly by `ExprId`, which is
/// sized by the *pool*: every pass allocated and zeroed one cell per node ever
/// interned, reachable or not, so a 7-node simplification cost 42 ms in a pool
/// of 8 million nodes.  Cells are now assigned densely as the traversal
/// reaches nodes, so the table is sized by the expression.  The map is only
/// written while the levels are built and only read afterwards, so workers
/// share it without synchronisation.
struct Table {
    slot: IdMap<Slot>,
    cells: Vec<AtomicU32>,
}

/// Where a reachable node's cell lives, and its height in the level order.
#[derive(Clone, Copy)]
struct Slot {
    cell: u32,
    height: u32,
}

impl Table {
    fn cell(&self, id: ExprId) -> &AtomicU32 {
        &self.cells[self.slot[&id].cell as usize]
    }
}

/// Rebuild `id` with each child replaced by its mapped value, reusing `id`
/// itself when nothing changed.
///
/// `ExprPool::add` and `ExprPool::mul` canonically sort their arguments, so the
/// reuse shortcut for those is only taken when the original list is already
/// sorted — otherwise reusing `id` would skip a canonicalisation.
fn rebuild(id: ExprId, data: &ExprData, pool: &ExprPool, table: &Table) -> ExprId {
    let map = |c: ExprId| lookup(table, c);
    match data {
        ExprData::Add(args) => {
            let new: Vec<ExprId> = args.iter().map(|&a| map(a)).collect();
            if new == *args && is_sorted(args) {
                id
            } else {
                pool.add(new)
            }
        }
        ExprData::Mul(args) => {
            let new: Vec<ExprId> = args.iter().map(|&a| map(a)).collect();
            if new == *args && is_sorted(args) {
                id
            } else {
                pool.mul(new)
            }
        }
        ExprData::Pow { base, exp } => {
            let (b, e) = (map(*base), map(*exp));
            if b == *base && e == *exp {
                id
            } else {
                pool.pow(b, e)
            }
        }
        ExprData::Func { name, args } => {
            let new: Vec<ExprId> = args.iter().map(|&a| map(a)).collect();
            if new == *args {
                id
            } else {
                pool.func(name.as_str(), new)
            }
        }
        ExprData::Piecewise { branches, default } => {
            // Conditions are passed through unchanged, as in `simplify_children`.
            let new: Vec<(ExprId, ExprId)> = branches.iter().map(|&(c, v)| (c, map(v))).collect();
            let d = map(*default);
            if new == *branches && d == *default {
                id
            } else {
                pool.piecewise(new, d)
            }
        }
        ExprData::Predicate { kind, args } => {
            let new: Vec<ExprId> = args.iter().map(|&a| map(a)).collect();
            if new == *args {
                id
            } else {
                pool.predicate(kind.clone(), new)
            }
        }
        ExprData::Forall { var, body } => {
            let b = map(*body);
            if b == *body {
                id
            } else {
                pool.forall(*var, b)
            }
        }
        ExprData::Exists { var, body } => {
            let b = map(*body);
            if b == *body {
                id
            } else {
                pool.exists(*var, b)
            }
        }
        ExprData::BigO(arg) => {
            let a = map(*arg);
            if a == *arg {
                id
            } else {
                pool.big_o(a)
            }
        }
        // Atoms have no children; `id` already interns this exact node.
        _ => id,
    }
}

fn is_sorted(args: &[ExprId]) -> bool {
    args.windows(2).all(|w| w[0] <= w[1])
}

// ---------------------------------------------------------------------------
// Level construction
// ---------------------------------------------------------------------------

/// Bucket every node reachable from `root` by height, so that `levels[h]`
/// contains only nodes whose children all live in `levels[..h]`.
///
/// Iterative post-order: a node is emitted after its children, so its height is
/// one more than the tallest child.  Shared subexpressions are visited once,
/// which is what makes each level a set of *independent* redexes.
///
/// Also returns the pass's substitution [`Table`], with one unwritten cell per
/// node reached.  Every side table here is keyed by the nodes the traversal
/// reaches, never by the pool's size — see [`Table`].
fn build_levels(root: ExprId, pool: &ExprPool) -> (Vec<Vec<ExprId>>, Table) {
    // `height` is `u32::MAX` from the moment a node is pushed until it is
    // emitted; a node absent from the map has not been reached yet.
    let mut slot: IdMap<Slot> = IdMap::default();
    let mut levels: Vec<Vec<ExprId>> = Vec::new();
    let mut stack: Vec<(ExprId, bool)> = vec![(root, false)];

    while let Some((id, expanded)) = stack.pop() {
        if expanded {
            let h = pool.with(id, |data| {
                let mut h = 0_u32;
                for_each_child(data, |c| {
                    let ch = slot[&c].height;
                    debug_assert_ne!(ch, u32::MAX, "child emitted after its parent");
                    h = h.max(ch.saturating_add(1));
                });
                h
            });
            slot.get_mut(&id).expect("pushed before emitted").height = h;
            let level = h as usize;
            if levels.len() <= level {
                levels.resize_with(level + 1, Vec::new);
            }
            levels[level].push(id);
        } else {
            // Lossless: the keys are `ExprId`s (a `u32`), so there are at
            // most 2^32 of them and `slot.len()` is below 2^32 before insert.
            let next = slot.len() as u32;
            if let std::collections::hash_map::Entry::Vacant(e) = slot.entry(id) {
                e.insert(Slot {
                    cell: next,
                    height: u32::MAX,
                });
            } else {
                continue;
            }
            stack.push((id, true));
            pool.with(id, |data| {
                for_each_child(data, |c| {
                    if !slot.contains_key(&c) {
                        stack.push((c, false));
                    }
                })
            });
        }
    }

    let cells = (0..slot.len()).map(|_| AtomicU32::new(UNMAPPED)).collect();
    (levels, Table { slot, cells })
}

/// Visit the children a rewrite pass descends into.
///
/// Deliberately mirrors `simplify_children`: piecewise *conditions* are not
/// descended into, and atoms (including `RootSum`) have no children.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;
    use crate::simplify::simplify;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn matches_sequential_on_wide_add() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let args: Vec<ExprId> = (0..64)
            .map(|i| {
                let x = pool.symbol(format!("x{i}"), Domain::Real);
                pool.add(vec![x, zero])
            })
            .collect();
        let expr = pool.add(args);
        assert_eq!(
            simplify(expr, &pool).value,
            simplify_redex(expr, &pool).value
        );
    }

    #[test]
    fn matches_sequential_on_shared_dag() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let zero = pool.integer(0_i32);
        let mut shared = x;
        for _ in 0..8 {
            shared = pool.mul(vec![shared, one]);
            shared = pool.add(vec![shared, zero]);
        }
        let args: Vec<ExprId> = (1..=64)
            .map(|i| {
                let c = pool.integer(i);
                pool.mul(vec![shared, c])
            })
            .collect();
        let expr = pool.add(args);
        let seq = simplify(expr, &pool);
        let red = simplify_redex(expr, &pool);
        assert_eq!(seq.value, red.value);
        assert_eq!(seq.log.len(), red.log.len());
    }

    #[test]
    fn constant_folds() {
        let pool = p();
        let args: Vec<ExprId> = (1..=20).map(|i| pool.integer(i)).collect();
        let expr = pool.add(args);
        let expected = pool.integer(210_i32);
        assert_eq!(simplify_redex(expr, &pool).value, expected);
    }

    /// The redex engine reports a declined expansion the same way
    /// `simplify_with` does — the bound is a fact about the pass, not about
    /// which engine ran it.
    #[test]
    fn expand_limit_is_recorded_by_the_redex_engine() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let expr = pool.pow(pool.add(vec![x, y, z]), pool.integer(9_i32));
        let config = SimplifyConfig {
            expand: true,
            ..Default::default()
        };
        let r = simplify_redex_with_config(expr, &pool, &config);
        assert_eq!(r.value, expr);
        assert!(r
            .log
            .steps()
            .iter()
            .any(|s| s.rule_name == crate::simplify::rules::EXPAND_POW_LIMIT_RULE));
    }

    #[test]
    fn expand_matches_sequential() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let two = pool.integer(2_i32);
        let sum = pool.add(vec![x, y]);
        let sq = pool.pow(sum, two);
        let config = SimplifyConfig {
            expand: true,
            ..Default::default()
        };
        let seq =
            crate::simplify::simplify_with(sq, &pool, &rules_for_config(&config), config.clone());
        let red = simplify_redex_with_config(sq, &pool, &config);
        assert_eq!(seq.value, red.value);
    }

    #[test]
    fn honours_static_domains() {
        let pool = p();
        let x = pool.symbol("x", Domain::Positive);
        let two = pool.integer(2_i32);
        let sq = pool.pow(x, two);
        let sqrt = pool.func("sqrt", vec![sq]);
        assert_eq!(
            simplify(sqrt, &pool).value,
            simplify_redex(sqrt, &pool).value
        );
    }

    /// The traversal is iterative, so depth is bounded by memory rather than by
    /// the stack — including on a rayon worker, where the stack is 2 MiB.
    #[test]
    fn deep_chain_does_not_recurse() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let zero = pool.integer(0_i32);
        let mut deep = x;
        for _ in 0..1000 {
            deep = pool.mul(vec![deep, one]);
            deep = pool.add(vec![deep, zero]);
            deep = pool.pow(deep, one);
        }
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        assert_eq!(tp.install(|| simplify_redex(deep, &pool).value), x);
    }

    #[test]
    fn log_is_deterministic_across_thread_counts() {
        let pool = p();
        let zero = pool.integer(0_i32);
        let one = pool.integer(1_i32);
        let args: Vec<ExprId> = (0..128)
            .map(|i| {
                let x = pool.symbol(format!("x{i}"), Domain::Real);
                let a = pool.mul(vec![x, one]);
                pool.add(vec![a, zero])
            })
            .collect();
        let expr = pool.add(args);

        let mut logs = Vec::new();
        for threads in [1usize, 4, 16] {
            let tp = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let r = tp.install(|| simplify_redex(expr, &pool));
            logs.push(
                r.log
                    .0
                    .iter()
                    .map(|s| (s.rule_name, s.before, s.after))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(logs[0], logs[1]);
        assert_eq!(logs[1], logs[2]);
    }

    /// The substitution table has one cell per node the pass can reach,
    /// however large the pool around the expression has grown.  It used to
    /// have one per pool node, so simplifying 7 nodes in a pool of 8 million
    /// allocated and zeroed an 8-million-cell table per pass (42 ms).
    #[test]
    fn table_is_sized_by_the_expression_not_the_pool() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let mut junk = x;
        for _ in 0..50_000 {
            junk = pool.mul(vec![pool.add(vec![junk, y]), x]);
        }
        let zero = pool.integer(0_i32);
        // `pool.add` splices the inner sum: x·x + y + 0 has x, y, 0, x·x, root.
        let expr = pool.add(vec![pool.mul(vec![x, x]), pool.add(vec![y, zero])]);
        assert!(pool.len() > 100_000);

        let (levels, table) = build_levels(expr, &pool);
        assert_eq!(table.cells.len(), 5);
        assert_eq!(levels.iter().map(Vec::len).sum::<usize>(), 5);
        assert_eq!(
            simplify_redex(expr, &pool).value,
            simplify(expr, &pool).value
        );
    }

    /// A shared node gets one cell and one slot in one level.
    #[test]
    fn shared_nodes_get_one_cell() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let minus_one = pool.integer(-1_i32);
        let (mut a, mut b) = (pool.integer(1_i32), x);
        for _ in 1..30 {
            let c = pool.add(vec![
                pool.mul(vec![two, x, b]),
                pool.mul(vec![minus_one, a]),
            ]);
            a = b;
            b = c;
        }
        let (levels, table) = build_levels(b, &pool);
        let placed: usize = levels.iter().map(Vec::len).sum();
        assert_eq!(table.cells.len(), placed);
        // 29 `Add`s, 29 `2·x·T` and 29 `−1·T` products (T₁ = x needs no
        // product of its own beyond the shared leaves), plus x, 1, 2, −1.
        assert!(placed < 100, "{placed} cells for a ~90-node DAG");
    }

    use crate::simplify::proptests::{build_dag, dag_ops};
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// The level scheduler, now indexed by reachable node rather than by
        /// `ExprId`, still agrees with the sequential engine on shared DAGs.
        #[test]
        fn matches_sequential_on_random_dags(ops in dag_ops(14)) {
            let pool = p();
            let e = build_dag(&pool, &ops, false);
            prop_assert_eq!(simplify_redex(e, &pool).value, simplify(e, &pool).value);
        }

        /// The dense index mapping: every reached node gets exactly one cell,
        /// the cells are exactly `0..n` (so `cells[cell]` is always in bounds),
        /// each node sits in one level at its recorded height, and all of its
        /// children sit in strictly lower levels.
        #[test]
        fn table_cells_are_a_dense_bijection(ops in dag_ops(14)) {
            let pool = p();
            let e = build_dag(&pool, &ops, false);
            let (levels, table) = build_levels(e, &pool);
            let n = table.slot.len();
            prop_assert_eq!(table.cells.len(), n);
            prop_assert_eq!(levels.iter().map(Vec::len).sum::<usize>(), n);
            let mut cells: Vec<u32> = table.slot.values().map(|s| s.cell).collect();
            cells.sort_unstable();
            prop_assert!(cells.iter().copied().eq(0..n as u32));
            for (h, level) in levels.iter().enumerate() {
                for &id in level {
                    prop_assert_eq!(table.slot[&id].height as usize, h);
                    let mut child_ok = true;
                    pool.with(id, |d| for_each_child(d, |c| {
                        child_ok &= (table.slot[&c].height as usize) < h;
                    }));
                    prop_assert!(child_ok);
                }
            }
        }
    }
}
