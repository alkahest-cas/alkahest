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
//! # Shared subexpressions
//!
//! The traversal is memoised through a concurrent `DashMap<ExprId, ExprId>`,
//! one per fixed-point pass, mirroring the `HashMap` memo in
//! [`crate::simplify::engine::simplify_with`].  Without it every occurrence of
//! a shared subexpression is re-simplified: because the pool is hash-consed,
//! ordinary expressions are DAGs rather than trees, and the duplicated work
//! made `simplify_par` orders of magnitude slower than the sequential path on
//! exactly the inputs it was supposed to accelerate.
//!
//! Two threads may still race to simplify the same node; both compute the same
//! value, so only the work (and possibly a duplicated derivation-log entry) is
//! wasted.  The memo is deliberately not held across the recursive call — doing
//! so would hold a `DashMap` shard lock while rayon work-stealing runs nested
//! tasks on the same thread.
//!
//! # Deep expressions
//!
//! The traversal is recursive, and rayon workers get the default 2 MiB stack
//! rather than the main thread's 8 MiB, so a deep chain used to abort the whole
//! process with a stack overflow.  The recursion now measures how much stack it
//! has consumed and continues on a freshly spawned thread with a larger stack
//! before running out; see [`with_stack_segment`].
//!
//! # Safety
//!
//! `ExprPool: Send + Sync` is asserted in `pool.rs`.  Reads go through
//! `ExprPool::with`, which borrows a node without cloning it; the node array is
//! append-only and reference-stable, so interning while a borrow is live is
//! sound.

#![cfg(feature = "parallel")]

use crate::deriv::log::{DerivationLog, DerivedExpr};
use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::simplify::engine::SimplifyConfig;
use crate::simplify::rules::RewriteRule;
use dashmap::DashMap;
use rayon::prelude::*;
use std::sync::Arc;

/// Arity threshold above which children are simplified in parallel.
const PAR_THRESHOLD: usize = 4;

/// Stack size handed to each refill thread (see [`with_stack_segment`]).
const SEGMENT_STACK_BYTES: usize = 16 * 1024 * 1024;

/// Stack the traversal may consume on a thread it did not create.  Rayon
/// workers get 2 MiB by default, so this leaves a wide margin for the frames
/// already below us and for whatever the rule engine needs.
const FOREIGN_STACK_BUDGET: usize = 512 * 1024;

/// Stack the traversal may consume on a segment thread it created itself.
/// Kept well under [`SEGMENT_STACK_BYTES`] so a single node visit can never
/// straddle the end of the segment.
const OWNED_STACK_BUDGET: usize = SEGMENT_STACK_BYTES - 4 * 1024 * 1024;

thread_local! {
    /// Stack address at which this thread's segment began; 0 until first probe.
    static SEGMENT_BASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// How much stack this thread's segment is allowed to consume.
    static SEGMENT_BUDGET: std::cell::Cell<usize> =
        const { std::cell::Cell::new(FOREIGN_STACK_BUDGET) };
}

/// Per-pass memo: input `ExprId` → simplified `ExprId`.
type Memo = DashMap<ExprId, ExprId>;

/// Shared rule list handed to every worker.
///
/// `RewriteRule` requires `Send + Sync`, so `Box<dyn RewriteRule>` is already
/// shareable; spelling the auto traits again would only prevent this list from
/// being passed to the shared rule loop in `engine`.
type Rules = Arc<Vec<Box<dyn RewriteRule>>>;

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
    let rules: Rules = Arc::new(crate::simplify::rules_for_config(config));
    let mut current = expr;
    let mut full_log = DerivationLog::new();
    for _ in 0..config.max_iterations {
        // Fresh memo per pass, exactly as `simplify_with` does, so each sweep
        // sees the expression produced by the previous one.
        let memo = Memo::new();
        let result = simplify_node_par(current, pool, &rules, &memo);
        full_log = full_log.merge(result.log);
        if result.value == current {
            break;
        }
        current = result.value;
    }

    // Assumption-driven (colored e-graph) pass, mirroring `simplify_with`.
    // Without this, `simplify_par` silently ignored both explicit assumptions
    // and static symbol domains that the sequential path honours.
    let mut assumptions = config.assumptions.clone();
    super::assumptions::collect_static_domain_facts(current, pool, &mut assumptions);
    if !assumptions.is_empty() {
        let colored = super::colored_egraph::apply_colored_if_needed(current, pool, &assumptions);
        return DerivedExpr::with_log(colored.value, full_log.merge(colored.log));
    }
    DerivedExpr::with_log(current, full_log)
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

fn simplify_node_par(
    expr: ExprId,
    pool: &ExprPool,
    rules: &Rules,
    memo: &Memo,
) -> DerivedExpr<ExprId> {
    // Shared-subexpression cache: a hit returns an empty log so the same
    // rewrite is not reported once per occurrence (as in `simplify_node`).
    if let Some(cached) = memo.get(&expr) {
        return DerivedExpr::new(*cached);
    }

    let result = with_stack_segment(|| {
        // `with` borrows the node instead of cloning it: the owning `ExprData`
        // clone was one heap allocation (and, for `Func`, one `String` clone)
        // per node visit on every worker.
        let (rebuilt, child_log) = pool.with(expr, |data| {
            simplify_children_par(expr, data, pool, rules, memo)
        });

        let (current, rule_log) =
            crate::simplify::engine::apply_rules(rebuilt, pool, rules.as_ref());
        DerivedExpr::with_log(current, child_log.merge(rule_log))
    });

    memo.insert(expr, result.value);
    result
}

/// Run `f`, moving it to a thread with a fresh [`SEGMENT_STACK_BYTES`] stack
/// once the current segment has consumed its budget.
///
/// Rayon workers have the default 2 MiB stack, which a deep expression chain
/// exhausts — and a stack overflow aborts the process rather than unwinding, so
/// the caller cannot catch it.  Depth alone is a poor proxy for stack use (debug
/// frames are several times larger than release ones), so this measures the
/// stack actually consumed since the segment began.  Segments are spawned
/// scoped, so the borrowed pool, rules and memo stay valid.
///
/// Inside a segment the ambient rayon pool is no longer installed, so nested
/// `par_iter` calls fall back to the global pool.  That only affects subtrees
/// deep enough to exhaust a whole segment, where returning an answer at all
/// matters more than which pool schedules the work.
fn with_stack_segment<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if stack_used() < SEGMENT_BUDGET.with(|b| b.get()) {
        return f();
    }
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .stack_size(SEGMENT_STACK_BYTES)
            .spawn_scoped(scope, || {
                // Fresh thread: its own thread-locals, and a stack we sized.
                SEGMENT_BUDGET.with(|b| b.set(OWNED_STACK_BUDGET));
                f()
            })
            .expect("failed to spawn stack segment for deep recursion")
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
    })
}

/// Stack bytes consumed on this thread since the current segment began.
///
/// Uses the address of a local as a stack-depth probe.  Stacks grow downwards
/// on every platform this crate targets, so a *smaller* address means deeper.
///
/// The baseline is re-established whenever the probe lands at or above it.
/// That matters because Rayon reuses its workers: the baseline used to be
/// latched on a thread's first probe and never revisited, so a worker that
/// happened to take its first `simplify_par` task from deep inside a call
/// chain kept that deep address as its baseline forever.  Every later task on
/// that worker started *above* the stale baseline, `saturating_sub` floored
/// the difference at 0, and the traversal read its own stack usage as zero no
/// matter how deep it went — so it never refilled, and ran off the end of the
/// worker's 2 MiB stack.  A stack overflow aborts the process, which is
/// precisely what this machinery exists to prevent.
///
/// Re-baselining upwards is always safe: an address above the current
/// baseline means the frames that baseline was measured against have already
/// returned, so it describes a stack that no longer exists.
fn stack_used() -> usize {
    let probe = 0u8;
    let here = &probe as *const u8 as usize;
    SEGMENT_BASE.with(|base| {
        if base.get() == 0 || here >= base.get() {
            base.set(here);
            0
        } else {
            base.get() - here
        }
    })
}

fn simplify_children_par(
    expr: ExprId,
    data: &ExprData,
    pool: &ExprPool,
    rules: &Rules,
    memo: &Memo,
) -> (ExprId, DerivationLog) {
    match data {
        ExprData::Add(args) if args.len() >= PAR_THRESHOLD => {
            let (new_args, log) = par_children(args, pool, rules, memo);
            (rebuild_nary(expr, args, new_args, pool, NAry::Add), log)
        }
        ExprData::Mul(args) if args.len() >= PAR_THRESHOLD => {
            let (new_args, log) = par_children(args, pool, rules, memo);
            (rebuild_nary(expr, args, new_args, pool, NAry::Mul), log)
        }
        // Sequential fallback for small nodes and Pow/Func
        ExprData::Add(args) => {
            let (new_args, log) = seq_children(args, pool, rules, memo);
            (rebuild_nary(expr, args, new_args, pool, NAry::Add), log)
        }
        ExprData::Mul(args) => {
            let (new_args, log) = seq_children(args, pool, rules, memo);
            (rebuild_nary(expr, args, new_args, pool, NAry::Mul), log)
        }
        ExprData::Pow { base, exp } => {
            let rb = simplify_node_par(*base, pool, rules, memo);
            let re = simplify_node_par(*exp, pool, rules, memo);
            let log = rb.log.merge(re.log);
            let id = if rb.value == *base && re.value == *exp {
                expr
            } else {
                pool.pow(rb.value, re.value)
            };
            (id, log)
        }
        ExprData::Func { name, args } => {
            let (new_args, log) = seq_children(args, pool, rules, memo);
            let id = if new_args == *args {
                expr
            } else {
                pool.func(name.as_str(), new_args)
            };
            (id, log)
        }
        ExprData::Piecewise { branches, default } => {
            let mut log = DerivationLog::new();
            let mut changed = false;
            let new_branches: Vec<(ExprId, ExprId)> = branches
                .iter()
                .map(|&(cond, val)| {
                    let rv = simplify_node_par(val, pool, rules, memo);
                    log = std::mem::take(&mut log).merge(rv.log);
                    changed |= rv.value != val;
                    (cond, rv.value)
                })
                .collect();
            let rd = simplify_node_par(*default, pool, rules, memo);
            log = log.merge(rd.log);
            let id = if !changed && rd.value == *default {
                expr
            } else {
                pool.piecewise(new_branches, rd.value)
            };
            (id, log)
        }
        ExprData::Predicate { kind, args } => {
            let (new_args, log) = seq_children(args, pool, rules, memo);
            let id = if new_args == *args {
                expr
            } else {
                pool.predicate(kind.clone(), new_args)
            };
            (id, log)
        }
        ExprData::Forall { var, body } => {
            let rb = simplify_node_par(*body, pool, rules, memo);
            let id = if rb.value == *body {
                expr
            } else {
                pool.forall(*var, rb.value)
            };
            (id, rb.log)
        }
        ExprData::Exists { var, body } => {
            let rb = simplify_node_par(*body, pool, rules, memo);
            let id = if rb.value == *body {
                expr
            } else {
                pool.exists(*var, rb.value)
            };
            (id, rb.log)
        }
        ExprData::BigO(arg) => {
            let r = simplify_node_par(*arg, pool, rules, memo);
            let id = if r.value == *arg {
                expr
            } else {
                pool.big_o(r.value)
            };
            (id, r.log)
        }
        // Atoms have no children; `expr` already interns this exact node.
        _ => (expr, DerivationLog::new()),
    }
}

/// Simplify `args` on the rayon pool, returning the new ids and merged log.
fn par_children(
    args: &[ExprId],
    pool: &ExprPool,
    rules: &Rules,
    memo: &Memo,
) -> (Vec<ExprId>, DerivationLog) {
    let results: Vec<DerivedExpr<ExprId>> = args
        .par_iter()
        .map(|&a| simplify_node_par(a, pool, rules, memo))
        .collect();
    let new_args: Vec<ExprId> = results.iter().map(|r| r.value).collect();
    let mut log = DerivationLog::new();
    for r in results {
        log = log.merge(r.log);
    }
    (new_args, log)
}

/// Simplify `args` in order on the current thread.
fn seq_children(
    args: &[ExprId],
    pool: &ExprPool,
    rules: &Rules,
    memo: &Memo,
) -> (Vec<ExprId>, DerivationLog) {
    let mut log = DerivationLog::new();
    let new_args: Vec<ExprId> = args
        .iter()
        .map(|&a| {
            let r = simplify_node_par(a, pool, rules, memo);
            log = std::mem::take(&mut log).merge(r.log);
            r.value
        })
        .collect();
    (new_args, log)
}

enum NAry {
    Add,
    Mul,
}

/// Re-intern an `Add`/`Mul` node, reusing `expr` when nothing changed.
///
/// `ExprPool::add` and `ExprPool::mul` canonically sort their arguments, so the
/// shortcut is only sound when the original list is already sorted — otherwise
/// reusing `expr` would skip a canonicalisation the rebuild would have applied.
fn rebuild_nary(
    expr: ExprId,
    old_args: &[ExprId],
    new_args: Vec<ExprId>,
    pool: &ExprPool,
    kind: NAry,
) -> ExprId {
    if new_args == old_args && old_args.windows(2).all(|w| w[0] <= w[1]) {
        return expr;
    }
    match kind {
        NAry::Add => pool.add(new_args),
        NAry::Mul => pool.mul(new_args),
    }
}

// ---------------------------------------------------------------------------
// Send + Sync rule list for parallel dispatch
// ---------------------------------------------------------------------------

/// Build a rule list where each rule is `Send + Sync`.
///
/// `RewriteRule` already requires `Send + Sync`, so this simply delegates to
/// [`crate::simplify::engine::rules_for_config`] rather than maintaining a
/// second hand-written list.  The duplicate list had silently drifted: it was
/// missing `ExpandPow`, so `simplify_par` with `expand: true` left `(x + y)^2`
/// unexpanded while `simplify` returned `x² + y² + 2·x·y`.
pub fn rules_for_config_par(config: &SimplifyConfig) -> Vec<Box<dyn RewriteRule + Send + Sync>> {
    crate::simplify::engine::rules_for_config(config)
        .into_iter()
        .map(|rule| rule as Box<dyn RewriteRule + Send + Sync>)
        .collect()
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

    /// The parallel rule list used to be a hand-maintained copy that had
    /// drifted from `rules_for_config` (it was missing `ExpandPow`).  Order
    /// matters: the rule loop fires the first match.
    #[test]
    fn ruleset_matches_sequential_exactly() {
        for expand in [false, true] {
            let config = SimplifyConfig {
                expand,
                ..Default::default()
            };
            let seq: Vec<&str> = crate::simplify::rules_for_config(&config)
                .iter()
                .map(|r| r.name())
                .collect();
            let par: Vec<&str> = rules_for_config_par(&config)
                .iter()
                .map(|r| r.name())
                .collect();
            assert_eq!(seq, par, "rule lists diverged for expand={expand}");
        }
    }

    #[test]
    fn par_expand_matches_sequential() {
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
        let seq = crate::simplify::simplify_with(
            sq,
            &pool,
            &crate::simplify::rules_for_config(&config),
            config.clone(),
        );
        let par = simplify_par_with_config(sq, &pool, &config);
        assert_eq!(seq.value, par.value, "(x + y)^2 expanded differently");
        // The expansion must actually have happened.
        assert_ne!(par.value, sq);
    }

    /// Static symbol domains authorise conditional rewrites in the sequential
    /// path; the parallel path used to skip that pass entirely.
    #[test]
    fn par_honours_static_domains() {
        let pool = p();
        let x = pool.symbol("x", Domain::Positive);
        let two = pool.integer(2_i32);
        let sq = pool.pow(x, two);
        let sqrt = pool.func("sqrt", vec![sq]);
        let seq = simplify(sqrt, &pool);
        let par = simplify_par(sqrt, &pool);
        assert_eq!(seq.value, par.value);
    }

    /// Shared subexpressions must be simplified once, not once per occurrence.
    /// Pinned to a single worker so the comparison is deterministic.
    #[test]
    fn par_memoises_shared_subexpressions() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let zero = pool.integer(0_i32);
        // A chunk of removable algebraic junk, shared by every term.
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
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let par = tp.install(|| simplify_par(expr, &pool));

        assert_eq!(seq.value, par.value);
        assert_eq!(
            seq.log.len(),
            par.log.len(),
            "parallel path re-simplified shared subexpressions"
        );
    }

    /// Build `((x * 1) + 0)^1` nested `depth` times — all of it removable.
    fn deep_chain(pool: &ExprPool, depth: usize) -> ExprId {
        let x = pool.symbol("x", Domain::Real);
        let one = pool.integer(1_i32);
        let zero = pool.integer(0_i32);
        let mut e = x;
        for _ in 0..depth {
            e = pool.mul(vec![e, one]);
            e = pool.add(vec![e, zero]);
            e = pool.pow(e, one);
        }
        e
    }

    /// A deep chain on a rayon worker (2 MiB stack) used to overflow the stack
    /// and abort the whole process instead of returning.  3000 recursion levels
    /// is roughly five times what an unguarded debug traversal survives there.
    ///
    /// Only the parallel path is exercised: the sequential traversal recurses
    /// just as deep and still overflows a small stack well before this size.
    #[test]
    fn par_survives_deep_chain_on_worker_thread() {
        let pool = p();
        let deep = deep_chain(&pool, 1000);
        let x = pool.symbol("x", Domain::Real);
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let par = tp.install(|| simplify_par(deep, &pool));
        assert_eq!(par.value, x);
    }

    /// Burn `frames` stack frames, then report `stack_used()` from the bottom.
    #[inline(never)]
    fn probe_at_depth(frames: u32) -> usize {
        // A real local keeps the frame from being optimised to nothing.
        let mut pad = [0u8; 256];
        pad[0] = frames as u8;
        std::hint::black_box(&pad);
        if frames == 0 {
            stack_used()
        } else {
            probe_at_depth(frames - 1)
        }
    }

    /// `SEGMENT_BASE` used to be latched on a thread's first probe and never
    /// revisited.  Rayon reuses its workers, so a worker whose first task
    /// probed from deep in a call chain kept that deep address forever; every
    /// later task started above it, `saturating_sub` floored the result at 0,
    /// and the traversal believed it was using no stack however deep it went.
    /// It therefore never refilled and eventually overflowed the worker's
    /// 2 MiB stack — an abort, not an error.
    ///
    /// Asserted here rather than by actually overflowing a stack: a
    /// regression must fail this test, not kill the test process.
    #[test]
    fn stack_probe_rebaselines_after_unwinding() {
        // Task 1: latch a baseline from deep in a call chain, then unwind.
        let deep = probe_at_depth(400);
        assert_eq!(deep, 0, "the first probe on a thread establishes the base");

        // Task 2 on the same (reused) thread, starting near the top of the
        // stack.  This must re-baseline...
        assert_eq!(stack_used(), 0, "a probe above the old base must re-base");

        // ...so that going deeper than *this* point is now measurable.  With
        // the stale baseline still in place this read 0, which is exactly the
        // under-read that let the traversal run off the end of the stack.
        let used = probe_at_depth(64);
        assert!(
            used > 0,
            "stack usage under-read as {used} after re-baselining"
        );
    }

    /// At a depth both paths can handle, the results must still agree.
    #[test]
    fn par_matches_sequential_on_moderate_chain() {
        let pool = p();
        let deep = deep_chain(&pool, 100);
        let seq = simplify(deep, &pool);
        let tp = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let par = tp.install(|| simplify_par(deep, &pool));
        assert_eq!(seq.value, par.value);
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
