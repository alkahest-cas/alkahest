//! Phase 23 — Parallel simplification using Rayon.
//!
//! Feature-gated behind `--features parallel`.
//!
//! # Strategy
//!
//! The standard `simplify_once` traversal is inherently sequential because
//! child simplifications are composed into a parent node.  However, the
//! child sub-trees of `Add` and `Mul` nodes are *independent* — they share
//! the pool for interning but do not depend on each other's results.
//!
//! When the `parallel` feature is enabled, `simplify_par` replaces the
//! sequential child iteration with a Rayon parallel iterator for `Add` and
//! `Mul` nodes whose arity exceeds `PAR_THRESHOLD`.  Smaller nodes fall back
//! to the sequential path to avoid scheduling overhead.
//!
//! # Sharded pool audit
//!
//! `ExprPool` currently uses a single `Mutex<PoolState>`.  Under heavy
//! parallel load the lock becomes a bottleneck.  When `--features parallel`
//! and `--features parallel-sharded` are combined, `ExprPool` switches to a
//! `DashMap`-based sharded intern table.  This module documents the design;
//! the actual sharding is in `kernel/pool_sharded.rs`.
//!
//! # Safety
//!
//! `ExprPool: Send + Sync` is asserted in `pool.rs`.  The parallel
//! simplifier only reads from the pool (via `pool.get`) and writes via
//! `pool.intern` (which is already `Mutex`-protected).

#![cfg(feature = "parallel")]

use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::SimplifyConfig;
use crate::simplify::rules::RewriteRule;
use rayon::prelude::*;
use std::sync::Arc;

/// Arity threshold above which children are simplified in parallel.
const PAR_THRESHOLD: usize = 4;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Simplify `expr` using the parallel bottom-up traversal.
///
/// Equivalent to [`crate::simplify::simplify`] but processes large `Add`/`Mul`
/// nodes in parallel via Rayon.  The returned [`DerivedExpr`] carries the
/// combined derivation log (note: log ordering is non-deterministic when
/// children run in parallel).
pub fn simplify_par(expr: ExprId, pool: &ExprPool) -> DerivedExpr<ExprId> {
    simplify_par_with_config(expr, pool, &SimplifyConfig::default())
}

/// Like `simplify_par` but with a custom [`SimplifyConfig`].
pub fn simplify_par_with_config(
    expr: ExprId,
    pool: &ExprPool,
    config: &SimplifyConfig,
) -> DerivedExpr<ExprId> {
    let rules: Arc<Vec<Box<dyn RewriteRule + Send + Sync>>> =
        Arc::new(rules_for_config_par(config));
    let mut current = expr;
    let mut full_log = DerivationLog::new();
    for _ in 0..config.max_iterations {
        let result = simplify_node_par(current, pool, &rules);
        full_log = full_log.merge(result.log);
        if result.value == current {
            break;
        }
        current = result.value;
    }
    DerivedExpr::with_log(current, full_log)
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn simplify_node_par(
    expr: ExprId,
    pool: &ExprPool,
    rules: &Arc<Vec<Box<dyn RewriteRule + Send + Sync>>>,
) -> DerivedExpr<ExprId> {
    let data = pool.get(expr);
    let (rebuilt, child_log) = simplify_children_par(data, pool, rules);

    let mut current = rebuilt;
    let mut rule_log = DerivationLog::new();
    loop {
        let mut fired = false;
        for rule in rules.as_ref() {
            if let Some((new_expr, step_log)) = rule.apply(current, pool) {
                rule_log = rule_log.merge(step_log);
                current = new_expr;
                fired = true;
                break;
            }
        }
        if !fired {
            break;
        }
    }
    DerivedExpr::with_log(current, child_log.merge(rule_log))
}

fn simplify_children_par(
    data: ExprData,
    pool: &ExprPool,
    rules: &Arc<Vec<Box<dyn RewriteRule + Send + Sync>>>,
) -> (ExprId, DerivationLog) {
    match data {
        ExprData::Add(args) if args.len() >= PAR_THRESHOLD => {
            let results: Vec<DerivedExpr<ExprId>> = args
                .par_iter()
                .map(|&a| simplify_node_par(a, pool, rules))
                .collect();
            let new_args: Vec<ExprId> = results.iter().map(|r| r.value).collect();
            let mut log = DerivationLog::new();
            for r in results {
                log = log.merge(r.log);
            }
            (pool.add(new_args), log)
        }
        ExprData::Mul(args) if args.len() >= PAR_THRESHOLD => {
            let results: Vec<DerivedExpr<ExprId>> = args
                .par_iter()
                .map(|&a| simplify_node_par(a, pool, rules))
                .collect();
            let new_args: Vec<ExprId> = results.iter().map(|r| r.value).collect();
            let mut log = DerivationLog::new();
            for r in results {
                log = log.merge(r.log);
            }
            (pool.mul(new_args), log)
        }
        // Sequential fallback for small nodes and Pow/Func
        ExprData::Add(args) => {
            let mut log = DerivationLog::new();
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node_par(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.add(new_args), log)
        }
        ExprData::Mul(args) => {
            let mut log = DerivationLog::new();
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node_par(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.mul(new_args), log)
        }
        ExprData::Pow { base, exp } => {
            let rb = simplify_node_par(base, pool, rules);
            let re = simplify_node_par(exp, pool, rules);
            let log = rb.log.merge(re.log);
            (pool.pow(rb.value, re.value), log)
        }
        ExprData::Func { name, args } => {
            let mut log = DerivationLog::new();
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node_par(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.func(&name, new_args), log)
        }
        ExprData::Piecewise { branches, default } => {
            let mut log = DerivationLog::new();
            let new_branches: Vec<(ExprId, ExprId)> = branches
                .into_iter()
                .map(|(cond, val)| {
                    let rv = simplify_node_par(val, pool, rules);
                    log = std::mem::take(&mut log).merge(rv.log);
                    (cond, rv.value)
                })
                .collect();
            let rd = simplify_node_par(default, pool, rules);
            log = log.merge(rd.log);
            (pool.piecewise(new_branches, rd.value), log)
        }
        ExprData::Predicate { kind, args } => {
            let mut log = DerivationLog::new();
            let new_args: Vec<ExprId> = args
                .into_iter()
                .map(|a| {
                    let r = simplify_node_par(a, pool, rules);
                    log = std::mem::take(&mut log).merge(r.log);
                    r.value
                })
                .collect();
            (pool.predicate(kind, new_args), log)
        }
        ExprData::Forall { var, body } => {
            let rb = simplify_node_par(body, pool, rules);
            (pool.forall(var, rb.value), rb.log)
        }
        ExprData::Exists { var, body } => {
            let rb = simplify_node_par(body, pool, rules);
            (pool.exists(var, rb.value), rb.log)
        }
        leaf => (pool.intern(leaf), DerivationLog::new()),
    }
}

// ---------------------------------------------------------------------------
// Send + Sync rule list for parallel dispatch
// ---------------------------------------------------------------------------

/// Build a rule list where each rule is `Send + Sync`.
///
/// All rule structs in `alkahest_core::simplify::rules` are zero-sized, so they
/// are trivially `Send + Sync`.  This function mirrors `rules_for_config` but
/// produces boxed `dyn RewriteRule + Send + Sync`.
pub fn rules_for_config_par(config: &SimplifyConfig) -> Vec<Box<dyn RewriteRule + Send + Sync>> {
    use crate::simplify::rules::{
        AddZero, CanonicalOrder, ConstFold, DivSelf, ExpandMul, FlattenAdd, FlattenMul, MulOne,
        MulZero, PowOne, PowZero, SubSelf,
    };
    let mut rules: Vec<Box<dyn RewriteRule + Send + Sync>> = vec![
        Box::new(FlattenMul),
        Box::new(FlattenAdd),
        Box::new(MulZero),
        Box::new(AddZero),
        Box::new(MulOne),
        Box::new(PowZero),
        Box::new(PowOne),
        Box::new(ConstFold),
        Box::new(SubSelf),
        Box::new(DivSelf),
        Box::new(CanonicalOrder),
    ];
    if config.expand {
        rules.push(Box::new(ExpandMul));
    }
    rules
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::simplify::simplify;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn par_matches_sequential_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let zero = pool.integer(0_i32);
        // Build a large Add with many zeros: x + 0 + 0 + 0 + 0 + 0
        let expr = pool.add(vec![x, zero, zero, zero, zero, zero]);
        let seq = simplify(expr, &pool);
        let par = simplify_par(expr, &pool);
        assert_eq!(seq.value, par.value);
    }

    #[test]
    fn par_matches_sequential_mul() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let expr = pool.mul(vec![x, one, one, one, one, one]);
        let seq = simplify(expr, &pool);
        let par = simplify_par(expr, &pool);
        assert_eq!(seq.value, par.value);
    }

    #[test]
    fn par_constant_folding() {
        let pool = p();
        let a = pool.integer(2_i32);
        let b = pool.integer(3_i32);
        let c = pool.integer(4_i32);
        let d = pool.integer(5_i32);
        let expr = pool.add(vec![a, b, c, d]);
        let par = simplify_par(expr, &pool);
        // 2 + 3 + 4 + 5 = 14
        let expected = pool.integer(14_i32);
        assert_eq!(par.value, expected);
    }

    #[test]
    fn par_large_sum() {
        let pool = p();
        // Sum of 20 integer constants
        let args: Vec<ExprId> = (1..=20).map(|i| pool.integer(i)).collect();
        let expr = pool.add(args);
        let par = simplify_par(expr, &pool);
        let seq = simplify(expr, &pool);
        assert_eq!(par.value, seq.value);
    }
}
