//! The expression-depth ceiling that keeps deep trees from killing the process.
//!
//! # Why this exists
//!
//! Almost every operation on an expression is a structural recursion over the
//! DAG: printing, differentiation, substitution, translation to Lean or
//! SMT-LIB, evaluation.  Each level of the expression costs one or more native
//! stack frames, and a native stack overflow is **not** an exception — the
//! kernel delivers `SIGSEGV` and the process dies with no traceback, no error
//! code, and nothing for a caller's `except Exception` to catch.  For an
//! unattended run that is strictly worse than a wrong answer: a wrong answer
//! can be logged.
//!
//! # What is bounded
//!
//! One number, [`MAX_EXPR_DEPTH`], against one quantity: the **cached node
//! depth** of an already-interned expression — the longest root-to-leaf path,
//! `1 + max(child depths)`, as returned by [`ExprPool::depth`].  Not stack
//! bytes, not node count, not width.  A 50 000-term `Add` is depth 2 and is
//! accepted; a `Pow` tower of depth 2 049 is refused however few nodes it has.
//!
//! Nothing in this module measures the stack.  The one mechanism in this crate
//! that does is a different one with a different failure mode — see *What this
//! does not cover*.
//!
//! The number came from bisection on the shipped release build with the usual
//! 8 MiB main-thread stack (`ulimit -s 8192`), on a chain of `sin`
//! applications:
//!
//! | operation | deepest that returned | first that segfaulted |
//! |---|---|---|
//! | `symbolic_grad` (reverse-mode DFS) | 4 625 | 4 687 |
//! | `simplify`, `to_lean` | 9 216 | 9 472 |
//! | `latex` | 13 312 | 13 824 |
//! | `unicode_str` | 15 360 | 15 872 |
//! | `str` / `repr` | 23 552 | 24 576 |
//!
//! [`MAX_EXPR_DEPTH`] sits below the worst of those with room to spare, so one
//! number covers all of them instead of each walker carrying its own.  The
//! `simplify` row is history rather than a current measurement: that traversal
//! no longer overflows at any depth (below).  `symbolic_grad` is the shallowest
//! walker that still depends on this ceiling, and is what the number is
//! calibrated against.
//!
//! That calibration assumes the **shipped release build on an 8 MiB stack**,
//! which is what a Python caller gets on the main thread.  A debug build has
//! frames several times larger — one level of the simplifier was measured at
//! 10 832 bytes under debug + AddressSanitizer — and a `cargo test` worker or a
//! Rayon worker has a 2 MiB stack, so those configurations can overflow *below*
//! this limit.  A test that means to reach the cap should run on a thread it
//! sized itself.
//!
//! # How it is enforced, and by whom
//!
//! [`ExprPool`] caches each node's depth at intern time, so
//! [`check_expr_depth`] is a single array read and an integer compare.  That
//! matters: the guard sits on hot paths such as `__str__`, and anything that
//! had to walk the tree to find its depth would cost more than it saves.
//!
//! The callers are the PyO3 bindings, and only those.  `alkahest-py` calls
//! [`check_expr_depth`] — through its `guard_depth` / `guard_expr_depth`
//! wrappers — or [`check_expr_depths`] at upwards of sixty entry points: every
//! renderer, `diff`, `symbolic_grad`, `subs`, the evaluators and the JIT
//! entry, the polynomial converters, the integrator, the SMT-LIB and Lean
//! emitters, the plotters.  `tests/test_expression_depth_limit.py` pins that
//! list.  Nothing inside this crate calls either function.
//!
//! # What this does not cover
//!
//! * **Rust callers.** A downstream user of `alkahest_cas` never crosses the
//!   PyO3 boundary, so nothing checks their input on their behalf.  Calling
//!   [`check_expr_depth`] at their own entry points is up to them.
//! * **Expressions this crate builds.** The check happens on the way in.
//!   `diff`, expansion, `series` and the integrator all deepen an expression
//!   *after* it was checked, and nothing re-checks the result.
//! * **New bindings.** A `#[pyfunction]` added without a `guard_depth` call
//!   inherits nothing; the guard is a convention held up by that test file, not
//!   by a type.
//! * **The bottom-up simplification traversals**, which have something
//!   stronger.  `simplify::engine`'s `simplify_node` and
//!   `simplify_node_indexed`, and — under the `parallel` feature —
//!   `simplify::parallel`'s `simplify_node_par`, all run under the
//!   segmented-stack trampoline in `simplify::stack`, which continues the
//!   recursion on a freshly spawned, larger-stacked thread before the current
//!   one is spent.  For those the depth bound is removed rather than lowered:
//!   they are limited by how many threads the OS will hand out, not by any one
//!   stack, and they truncate nothing.  The trampoline itself is not
//!   feature-gated and is not confined to the parallel simplifier; the other
//!   simplification strategies (`redex`, the e-graph passes) do not use it.
//!   The ceiling here still applies to all of them at the PyO3 boundary, where
//!   for the trampolined ones it is now a courtesy to the caller rather than
//!   what keeps the process alive.
//!
//! The trampoline is also where this crate's one stack *measurement* lives, and
//! it is deliberately not what bounds that recursion.  The probe takes the
//! address of a local; under AddressSanitizer's stack-use-after-return
//! detection such locals are relocated into a per-thread fake-stack ring whose
//! addresses *ascend* with depth, so the probe reads 0 however deep the
//! traversal goes.  A governor that under-reads never refills, which is how the
//! nightly `asan` shard died; raising `RUST_MIN_STACK` could not have fixed it,
//! because a probe that reports 0 reports 0 at any stack size.  That trampoline
//! is therefore bounded by an exact count of recursion levels, with the byte
//! probe kept only as an advisory backstop.  None of it applies to
//! [`check_expr_depth`], which reads a cached integer and measures no stack at
//! all — under a sanitizer it behaves exactly as it does anywhere else.
//!
//! # Configuration
//!
//! There is none.  [`MAX_EXPR_DEPTH`] is a compile-time constant with no
//! environment variable, per-pool override or Python setting behind it.  The
//! recursive-descent parser's own ceiling is *defined* as equal to it — it has
//! to count separately, because the overflow it prevents happens before any
//! node is interned and so before there is a cached depth to read.
//!
//! # What a caller should do about a refusal
//!
//! [`DepthLimitError`] is a normal, catchable, coded error (`E-DEPTH-001`,
//! cause `Resource`).  `alkahest-py` maps it to `alkahest.DepthLimitError`,
//! which derives from `AlkahestError` and so from `ValueError`.  It is a
//! refusal, not a panic and not a silent decline: the call raises rather than
//! returning a partial answer, and the batch entry points (`simplify_many`,
//! `diff_many`) record it on the failing item instead of raising.  Rebuild the
//! expression with less nesting, or split the work into subexpressions.
//!
//! `Add` and `Mul` no longer contribute to this problem at all.  Both splice
//! nested same-operator children at construction (see [`ExprPool::add`] /
//! [`ExprPool::mul`]), so a balanced `Add` of 100 000 terms and the same terms
//! accumulated one at a time with `+` are now *the same* depth-2 node — the
//! left-associated chain that a parser or a `fold` produces can no longer trip
//! this ceiling.  What still nests, and so still has to be refused, is
//! everything else: `Pow` towers, `Func` chains such as `sin(sin(sin(…)))`, and
//! `Piecewise`/`RootSum` nesting.

use crate::errors::AlkahestError;
use crate::kernel::{ExprId, ExprPool};
use std::fmt;

/// Deepest expression the PyO3 entry points will accept.
///
/// Compared against the cached node depth ([`ExprPool::depth`]); see the module
/// documentation for the measurements behind the number and for the callers
/// that actually apply it.  The shallowest walker to fall over did so at depth
/// 4 687 on an 8 MiB stack, so this leaves a factor of ~2.3 for stacks that
/// already have frames on them, for debug builds (whose frames are several
/// times larger than release ones), and for future walkers that use more stack
/// per level than today's.
///
/// It is deliberately *one* number rather than a per-operation table: a caller
/// that gets `str(expr)` to work should not then be surprised by a segfault
/// from `symbolic_grad(expr)`.  It is not, however, automatic — a walker added
/// later is covered only once its entry point calls [`check_expr_depth`].
///
/// Compile-time only.  Nothing reads it from the environment or lets a caller
/// raise it, so a caller that needs deeper trees has to reshape them.
pub const MAX_EXPR_DEPTH: u32 = 2048;

/// An expression was too deeply nested to be processed by recursion.
///
/// Returned rather than risking a stack overflow; see the [module
/// documentation](self).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthLimitError {
    /// Depth of the offending expression, saturating at [`u32::MAX`].
    pub depth: u32,
    /// The ceiling that was exceeded — always [`MAX_EXPR_DEPTH`] today.
    pub limit: u32,
}

impl fmt::Display for DepthLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "expression nesting depth {} exceeds the limit of {}; \
             recursing over it would overflow the stack",
            self.depth, self.limit
        )
    }
}

impl std::error::Error for DepthLimitError {}

impl AlkahestError for DepthLimitError {
    fn code(&self) -> &'static str {
        "E-DEPTH-001"
    }

    fn remediation(&self) -> Option<&'static str> {
        Some("rebuild the expression with less nesting (a balanced n-ary Add is shallow where a chain of binary ones is not), or process it in smaller pieces")
    }
}

/// Refuse `id` if recursing over it would risk a stack overflow.
///
/// O(1) — compares the node depth cached when `id` was interned against
/// [`MAX_EXPR_DEPTH`].  It does not walk the expression and does not measure
/// the stack.
///
/// Call this at the entry point of anything that walks an expression
/// recursively: nothing calls it for you, and today every caller is in
/// `alkahest-py`.  See the [module documentation](self) for why, and for what
/// the check does not reach.
///
/// ```
/// use alkahest_cas::kernel::depth::{check_expr_depth, MAX_EXPR_DEPTH};
/// use alkahest_cas::kernel::{Domain, ExprPool};
///
/// let pool = ExprPool::new();
/// let x = pool.symbol("x", Domain::Real);
/// assert!(check_expr_depth(&pool, x).is_ok());
///
/// let mut deep = x;
/// for _ in 0..MAX_EXPR_DEPTH {
///     deep = pool.func("sin", vec![deep]);
/// }
/// let err = check_expr_depth(&pool, deep).unwrap_err();
/// assert_eq!(err.limit, MAX_EXPR_DEPTH);
/// ```
pub fn check_expr_depth(pool: &ExprPool, id: ExprId) -> Result<(), DepthLimitError> {
    let depth = pool.depth(id);
    if depth > MAX_EXPR_DEPTH {
        Err(DepthLimitError {
            depth,
            limit: MAX_EXPR_DEPTH,
        })
    } else {
        Ok(())
    }
}

/// Like [`check_expr_depth`] but for a batch of expressions.
///
/// Reports the first offender, so a caller handed a hundred expressions does
/// not have to find the bad one itself.
pub fn check_expr_depths(pool: &ExprPool, ids: &[ExprId]) -> Result<(), DepthLimitError> {
    for &id in ids {
        check_expr_depth(pool, id)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Domain;

    /// Depth is the *longest* root-to-leaf path, and hash-consing must not
    /// confuse it: `sin(x) + x` is 3 (Add → Func → Symbol), not 2.
    #[test]
    fn depth_is_the_longest_path_not_the_shortest() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        assert_eq!(pool.depth(x), 1);
        let s = pool.func("sin", vec![x]);
        assert_eq!(pool.depth(s), 2);
        let sum = pool.add(vec![s, x]);
        assert_eq!(pool.depth(sum), 3);
    }

    /// A wide expression is shallow; the guard must not confuse size with
    /// depth, or `check_expr_depth` would reject perfectly printable inputs.
    #[test]
    fn width_does_not_count_towards_depth() {
        let pool = ExprPool::new();
        let terms: Vec<_> = (0..10_000).map(|i| pool.integer(i)).collect();
        let wide = pool.add(terms);
        assert_eq!(pool.depth(wide), 2);
        assert!(check_expr_depth(&pool, wide).is_ok());
    }

    /// The same terms accumulated pairwise used to be deep — that was the shape
    /// that segfaulted every printer.  It is not deep any more: `add` splices
    /// nested `Add`s at construction, so a left-associated chain of `+` builds
    /// the very same flat, depth-2 node as adding all the terms at once.
    #[test]
    fn a_chain_of_binary_adds_is_flat_and_accepted() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut acc = x;
        for i in 0..MAX_EXPR_DEPTH {
            let k = pool.integer(i);
            acc = pool.add(vec![acc, k]);
        }
        assert_eq!(pool.depth(acc), 2, "a chain of binary adds must flatten");
        assert!(check_expr_depth(&pool, acc).is_ok());

        // …and it is literally the same node as the wide construction.
        let mut terms = vec![x];
        terms.extend((0..MAX_EXPR_DEPTH).map(|i| pool.integer(i)));
        assert_eq!(acc, pool.add(terms));
    }

    /// Depth still bites on the node kinds that genuinely nest.  `Pow` is the
    /// surviving stand-in for the binary-`add` chain above: nothing splices it,
    /// so one level of tower is one level of depth, and one past the ceiling is
    /// refused.
    #[test]
    fn a_chain_of_nesting_nodes_is_refused_past_the_limit() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let two = pool.integer(2_i32);
        let mut acc = x;
        for _ in 0..MAX_EXPR_DEPTH {
            acc = pool.pow(acc, two);
        }
        assert_eq!(pool.depth(acc), MAX_EXPR_DEPTH + 1);
        let err = check_expr_depth(&pool, acc).expect_err("one past the limit must be refused");
        assert_eq!(err.depth, MAX_EXPR_DEPTH + 1);
        assert_eq!(err.code(), "E-DEPTH-001");
    }

    /// Exactly at the limit is accepted — the boundary is inclusive, so the
    /// documented number is the deepest expression that still works.
    #[test]
    fn the_limit_itself_is_accepted() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut acc = x;
        for _ in 1..MAX_EXPR_DEPTH {
            acc = pool.func("sin", vec![acc]);
        }
        assert_eq!(pool.depth(acc), MAX_EXPR_DEPTH);
        assert!(check_expr_depth(&pool, acc).is_ok());
    }

    #[test]
    fn batch_check_reports_the_first_offender() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut deep = x;
        for _ in 0..=MAX_EXPR_DEPTH {
            deep = pool.func("sin", vec![deep]);
        }
        assert!(check_expr_depths(&pool, &[x, x]).is_ok());
        assert!(check_expr_depths(&pool, &[x, deep, x]).is_err());
    }
}
