//! The expression-depth ceiling that keeps deep trees from killing the process.
//!
//! # Why this exists
//!
//! Almost every operation on an expression is a structural recursion over the
//! DAG: printing, simplification, differentiation, substitution, translation to
//! Lean or SMT-LIB, evaluation.  Each level of the expression costs one or more
//! native stack frames, and a native stack overflow is **not** an exception —
//! the kernel delivers `SIGSEGV` and the process dies with no traceback, no
//! error code, and nothing for a caller's `except Exception` to catch.  For an
//! unattended run that is strictly worse than a wrong answer: a wrong answer
//! can be logged.
//!
//! Measured by bisection on the shipped release build with the usual 8 MiB
//! main-thread stack (`ulimit -s 8192`), on a chain of `sin` applications:
//!
//! | operation | deepest that returned | first that segfaulted |
//! |---|---|---|
//! | `symbolic_grad` (reverse-mode DFS) | 4 625 | 4 687 |
//! | `simplify`, `to_lean` | 9 216 | 9 472 |
//! | `latex` | 13 312 | 13 824 |
//! | `unicode_str` | 15 360 | 15 872 |
//! | `str` / `repr` | 23 552 | 24 576 |
//!
//! [`MAX_EXPR_DEPTH`] is set below the worst of those with room to spare, so
//! that every consumer refuses before any of them overflows, and one number
//! covers all of them instead of each walker carrying its own.
//!
//! The ceiling is calibrated for the **shipped release build on an 8 MiB
//! stack**, which is what a Python caller gets on the main thread.  A debug
//! build has frames several times larger, and a `cargo test` worker or a Rayon
//! worker has a 2 MiB stack, so those configurations can still overflow below
//! this limit; a test that means to reach the cap should run on a thread it
//! sized itself.  (`simplify_par` already handles the Rayon case by hopping to
//! a thread with a stack it sized itself — see `simplify::parallel`, which is
//! only compiled with the `parallel` feature.)
//!
//! # How it is enforced
//!
//! [`ExprPool`] caches each node's depth at intern time, so
//! [`check_expr_depth`] is a single array read and an integer compare.  That
//! matters: the guard sits on hot paths such as `__str__`, and anything that
//! had to walk the tree to find its depth would cost more than it saves.
//!
//! # What a caller should do about a refusal
//!
//! [`DepthLimitError`] is a normal, catchable, coded error (`E-DEPTH-001`).
//! Rebuild the expression with less nesting — a balanced `Add` of 100 000 terms
//! has depth 2, while the same terms accumulated one at a time with `+` have
//! depth 100 000 — or split the work into subexpressions.

use crate::errors::AlkahestError;
use crate::kernel::{ExprId, ExprPool};
use std::fmt;

/// Deepest expression any recursive consumer will accept.
///
/// See the module documentation for the measurements behind this number.  The
/// shallowest walker to fall over did so at depth 4 687 on an 8 MiB stack, so
/// this leaves a factor of ~2.3 for stacks that already have frames on them,
/// for debug builds (whose frames are several times larger than release ones),
/// and for future walkers that use more stack per level than today's.
///
/// It is deliberately *one* number rather than a per-operation table: a caller
/// that gets `str(expr)` to work should not then be surprised by a segfault
/// from `symbolic_grad(expr)`, and a walker added later inherits the guard
/// instead of having to remember to measure itself.
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
/// O(1) — the depth was cached when `id` was interned.  Call this at the entry
/// point of anything that walks an expression recursively; see the [module
/// documentation](self) for why.
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

    /// The same terms accumulated pairwise are deep, and that is exactly the
    /// shape that used to segfault every printer.
    #[test]
    fn a_chain_of_binary_adds_is_refused_past_the_limit() {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let mut acc = x;
        for i in 0..MAX_EXPR_DEPTH {
            let k = pool.integer(i);
            acc = pool.add(vec![acc, k]);
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
